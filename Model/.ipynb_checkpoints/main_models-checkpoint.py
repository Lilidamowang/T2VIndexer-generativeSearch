import os
import pickle
import random
from time import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import hues
import json
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from main_utils import assert_all_frozen, load_data_infer, \
    load_data, numerical_decoder, dec_2d, load_data_diy, load_data_infer_diy
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

LAYERS = 1

class Node(object):
    def __init__(self, token_id) -> None:
        self.token_id = token_id
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class TreeBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        '''
        seq is List[Int] representing id, without leading pad_token(hardcoded 0), with trailing eos_token(hardcoded 1) and (possible) pad_token, every int is token index
        e.g: [ 9, 14, 27, 38, 47, 58, 62,  1,  0,  0,  0,  0,  0,  0,  0]
        '''
        cur = self.root
        for tok in seq:
            if tok == 0:  # reach pad_token
                return
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]


def encode_single_newid(args, seq):
    '''
    Param:
        seq: doc_id string to be encoded, like "23456"
    Return:
        List[Int]: encoded tokens
    '''
    target_id_int = []
    seq = str(seq)
    if args.kary:
        for i, c in enumerate(seq.split('-')):
            if args.position:
                cur_token = i * args.kary + int(c) + 2
            else:
                cur_token = int(c) + 2
            target_id_int.append(cur_token)
    else:
        for i, c in enumerate(seq):
            if args.position:
                cur_token = i * 10 + int(c) + 2  # hardcoded vocab_size = 10
            else:
                cur_token = int(c) + 2
            target_id_int.append(cur_token)
    return target_id_int + [1]  # append eos_token


def decode_token(args, seqs):
    '''
    Param:
        seqs: 2d ndarray to be decoded
    Return:
        doc_id string, List[str]
    '''
    result = []
    for seq in seqs:
        try:
            eos_idx = seq.tolist().index(1)
            seq = seq[1: eos_idx]
        except:
            print("no eos token found")
        if args.position:
            offset = np.arange(len(seq)) * args.output_vocab_size + 2
        else:
            offset = 2
        res = seq - offset
        #assert np.all(res >= 0)
        if args.kary:
            result.append('-'.join(str(c) for c in res))
        else:
            result.append(''.join(str(c) for c in res))
    return result

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class MainDataset(Dataset):
    def __init__(self, args, tokenizer, path,task="train"):
        self.args = args
        input_length = args.max_input_length  # 40
        output_length = args.max_output_length * int(np.log10(args.output_vocab_size))  # 10
        inf_input_length = args.inf_max_input_length  # 40
        random_gen = args.random_gen # 0
        softmax = args.softmax # 0
        aug = args.aug # 0

        if task == "train":
            self.dataset, self.q_emb, self.query_dict, \
                self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict = \
                    load_data_diy(args, path)
        elif task == "test":
            self.dataset = load_data_infer_diy(args,path)
            self.q_emb, self.query_dict, \
            self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict \
                = None, None, None, None, None
        
        self.task = task
        self.input_length = input_length
        self.doc_length = self.args.doc_length
        self.inf_input_length = inf_input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = False
        self.softmax = softmax
        self.aug = aug
        self.random_gen = random_gen

        self.random_min = 2
        self.random_max = 6
        self.vocabs = set(self.tokenizer.get_vocab().keys())
        for token in [self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.sep_token,
                      self.tokenizer.pad_token, self.tokenizer.cls_token,
                      self.tokenizer.mask_token] + tokenizer.additional_special_tokens:
            if token is not None:
                self.vocabs.remove(token)
    
    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def clean_text(text):
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text
    
    def convert_to_features(self, example_batch, length_constraint):
        # Tokenize contexts and questions (as pairs of inputs)

        input_ = self.clean_text(example_batch)
        output_ = self.tokenizer.batch_encode_plus([input_], max_length=length_constraint,
                                                  padding='max_length', truncation=True, return_tensors="pt")

        return output_
    
    def __getitem__(self, index):
        inputs = self.dataset[index]
        query_embedding = torch.tensor([0])
        prefix_embedding, prefix_mask = torch.tensor([0]), torch.tensor([0])

        query, target, rank, neg_target, aug_query = inputs[0], inputs[1], inputs[2], inputs[4], inputs[5]
        aug_query = np.random.choice(aug_query, 1)[0]
        source = self.convert_to_features(query, self.input_length if self.task=='train' else self.inf_input_length)
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        aug_source = self.convert_to_features(aug_query, self.input_length if self.task=='train' else self.inf_input_length)
        aug_source_ids = aug_source["input_ids"].squeeze()
        aug_source_mask = aug_source["attention_mask"].squeeze()
        targets = self.convert_to_features(target, self.output_length)
        target_ids = targets["input_ids"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        def target_to_prefix_emb(target, tgt_length):
            tgt_prefix_emb = []
            prefix_masks = []
            for i in range(tgt_length):
                if i < len(target):
                    ###### fake data
                    _prefix_emb = np.random.rand(10, 768)
                    ###### real data
                    # _prefix_emb = self.prefix_embedding[self.prefix2idx_dict[target[:i]]]
                    _prefix_emb = torch.tensor(_prefix_emb)
                    tgt_prefix_emb.append(_prefix_emb.unsqueeze(0))
                    ##############################
                    ###### fake data
                    _prefix_mask = np.random.rand(10,)
                    _prefix_mask[_prefix_mask < 0.5] = 0
                    _prefix_mask[_prefix_mask > 0.5] = 1
                    ###### real data
                    # _prefix_mask = self.prefix_mask[self.prefix2idx_dict[target[:i]]]
                    _prefix_mask = torch.LongTensor(_prefix_mask)
                    prefix_masks.append(_prefix_mask.unsqueeze(0))
                    ##############################
                else:
                    tgt_prefix_emb.append(torch.zeros((1, 10,768)))
                    prefix_masks.append(torch.zeros((1, 10)))
            return torch.cat(tgt_prefix_emb, dim=0), torch.cat(prefix_masks, dim=0)

        neg_target_ids_list = []
        neg_target_mask_list = []
        neg_rank_list = []

        lm_labels = torch.zeros(self.args.max_output_length, dtype=torch.long)

        def decode_embedding_process(target_ids):
            target_id = self.tokenizer.decode(target_ids)
            target_id_int = []
            if self.args.kary:
                idx = 0 
                target_id = target_id.split('-')
                for i in range(0, len(target_id)):
                    c = target_id[i]
                    if self.args.position:
                        temp = i * self.args.output_vocab_size + int(c) + 2 \
                            if not self.args.hierarchic_decode else int(c) + 2
                    else:
                        temp = int(c) + 2
                    target_id_int.append(temp)
            else:
                bits = int(np.log10(self.args.output_vocab_size))
                idx = 0
                for i in range(0, len(target_id), bits):
                    if i + bits >= len(target_id):
                        c = target_id[i:]
                    c = target_id[i:i + bits]
                    if self.args.position:
                        temp = idx * self.args.output_vocab_size + int(c) + 2 \
                            if not self.args.hierarchic_decode else int(c) + 2
                    else:
                        temp = int(c) + 2
                    target_id_int.append(temp)
                    idx += 1
            lm_labels[:len(target_id_int)] = torch.LongTensor(target_id_int)
            lm_labels[len(target_id_int)] = 1
            decoder_attention_mask = lm_labels.clone()
            decoder_attention_mask[decoder_attention_mask != 0] = 1
            target_ids = lm_labels
            target_mask = decoder_attention_mask
            return target_ids, target_mask

        target_ids, target_mask = decode_embedding_process(target_ids)

        # print("source_ids", source_ids)
        # print("src_mask", src_mask)
        # print("aug_source_ids", aug_source_ids)
        # print("aug_source_mask", aug_source_mask)
        # print("target_ids", target_ids)
        # print("target_mask", target_mask)
        # print("neg_target_ids_list", neg_target_ids_list)
        # print("neg_rank_list", neg_rank_list)
        # print("neg_target_mask_list", neg_target_mask_list)
        # print("rank", rank)
        # print("query_embedding", query_embedding)
        # print("prefix_embedding", prefix_embedding)
        # print("prefix_mask", prefix_mask);input('s')

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "aug_source_ids": aug_source_ids,
                "aug_source_mask": aug_source_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "neg_target_ids": neg_target_ids_list,
                "neg_rank": neg_rank_list,
                "neg_target_mask": neg_target_mask_list,
                "doc_ids": doc_ids if self.args.contrastive_variant != '' else torch.tensor([-1997], dtype=torch.int64),
                "doc_mask": doc_mask if self.args.contrastive_variant != '' else torch.tensor([-1997], dtype=torch.int64),
                "softmax_index": torch.tensor([inputs[-1]], dtype=torch.int64)
                                        if self.softmax else torch.tensor([-1997], dtype=torch.int64),
                "rank": rank,
                "query_emb":query_embedding,
                "prefix_emb":prefix_embedding,
                "prefix_mask":prefix_mask}

class T5FineTuner(pl.LightningModule):
    def __init__(self, args, train=True):
        super(T5FineTuner, self).__init__()
        print("Begin build tree")
        builder = TreeBuilder()
        train_file = f"../MSRVTTdataset/k{args.kary}_c{args.kary}_{args.info}/train.tsv"
        test_file = f"../MSRVTTdataset/k{args.kary}_c{args.kary}_{args.info}/test.tsv"
        self.test_file = test_file
        df_train = pd.read_csv(
                    train_file,
                    encoding='utf-8',
                    header=None, sep='\t')
        df_train.dropna(inplace=True) # 删除空值
        df_test = pd.read_csv(
                    test_file,
                    encoding='utf-8', 
                    header=None, sep='\t')
        df = pd.merge(df_train, df_test, how='outer') # 将训练集和测试集合并到一起
        ###########################################################
        #                         分层树构建                         #
        ###########################################################
        for _, (_, _, newid) in tqdm(df.iterrows()):
            toks = encode_single_newid(args, newid)   # 1为end token
            builder.add(toks)
        root = builder.build()
        self.root = root
        self.args = args
        self.save_hyperparameters(args)
        expand_scale = args.max_output_length if not args.hierarchic_decode else 1
        self.decode_vocab_size = args.output_vocab_size * expand_scale + 2
        ###########################################################
        #                         模型构建                         #
        ###########################################################
        t5_config = T5Config(
            num_layers=args.num_layers,
            num_decoder_layers=0 if args.softmax else args.num_decoder_layers,
            d_ff=args.d_ff,
            d_model=args.d_model,
            num_heads=args.num_heads,
            decoder_start_token_id=0,  # 1,
            output_past=True,
            d_kv=args.d_kv,
            dropout_rate=args.dropout_rate,
            decode_embedding=args.decode_embedding,
            hierarchic_decode=args.hierarchic_decode,
            decode_vocab_size=self.decode_vocab_size, 
            output_vocab_size=args.output_vocab_size,
            tie_word_embeddings=args.tie_word_embedding,
            tie_decode_embedding=args.tie_decode_embedding,
            contrastive=args.contrastive,
            Rdrop=args.Rdrop,
            Rdrop_only_decoder=args.Rdrop_only_decoder,
            Rdrop_loss=args.Rdrop_loss,
            adaptor_decode=args.adaptor_decode,
            adaptor_efficient=args.adaptor_efficient,
            adaptor_layer_num = args.adaptor_layer_num,
            embedding_distillation=args.embedding_distillation,
            weight_distillation=args.weight_distillation,
            input_dropout=args.input_dropout,
            denoising=args.denoising,
            multiple_decoder=args.multiple_decoder,
            decoder_num=args.decoder_num,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_output_length=args.max_output_length,
        )
        model = T5ForConditionalGeneration(t5_config) # 创建新的空模型，未进行预训练
        # 将预训练模型encoder的参数赋值给model
        pretrain_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        pretrain_params = dict(pretrain_model.named_parameters())
        for name, param in model.named_parameters():
            if name.startswith(("shared.", "encoder.")):
                with torch.no_grad():  # encoder部分不进行更新
                    param.copy_(pretrain_params[name])
        self.model = model
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = torch.nn.Softmax(dim=-1)

        train_dataset = MainDataset(self.args, self.tokenizer, path=train_file)
        self.l1_query_train_dataset = train_dataset
        self.t_total = (
                    (len(train_dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
                    // self.args.gradient_accumulation_steps
                    * float(self.args.num_train_epochs)
            )
    
    def forward(self, input_ids, aug_input_ids=None, encoder_outputs=None, attention_mask=None, aug_attention_mask=None, logit_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None, query_embedding=None, prefix_emb=None, prefix_mask=None, only_encoder=False, decoder_index=-1, input_mask=None):
        input_mask = None
        """
        aug_input_ids只增强后的数据，用于数据增强，相当于batch_size*2，删除也可以
        就是对原数据进行增强，也可以选择不增强
        """
        input_ids = torch.cat([input_ids, aug_input_ids.clone()], dim=0)
        attention_mask = torch.cat([attention_mask, aug_attention_mask], dim=0)
        decoder_attention_mask = torch.cat([decoder_attention_mask, decoder_attention_mask], dim=0)
        lm_labels = torch.cat([lm_labels, lm_labels], dim=0)

        out = self.model(
            input_ids,
            input_mask=input_mask,
            logit_mask=logit_mask,
            encoder_outputs=encoder_outputs,
            only_encoder=only_encoder,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
            query_embedding=query_embedding,
            prefix_embedding=prefix_emb,
            prefix_mask=prefix_mask,
            return_dict=True,
            output_hidden_states=True,
            decoder_index=decoder_index,
            loss_weight=None,
        )
        return out

    def _step_i(self, batch, i=-1, encoder_outputs=None, input_mask=None):
        lm_labels = batch["target_ids"]
        target_mask = batch['target_mask']
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        # print(type(batch)) # [dict]
        # print(lm_labels.shape) # [4, 10]   
        # print(target_mask.shape) # [4, 10]
        # print(len(batch))
        # input('s')

        outputs = self.forward(input_ids=batch["source_ids"], aug_input_ids=batch["aug_source_ids"],
                               attention_mask=batch["source_mask"], aug_attention_mask=batch["aug_source_mask"],
                               lm_labels=lm_labels, decoder_attention_mask=target_mask,
                               query_embedding=batch["query_emb"], decoder_index=i, encoder_outputs=encoder_outputs,
                               prefix_emb=batch["prefix_emb"], prefix_mask=batch["prefix_mask"], input_mask=input_mask)
        
        loss = outputs.loss

        orig_loss = outputs.orig_loss
        dist_loss = outputs.dist_loss
        q_emb_distill_loss = 0
        weight_distillation = 0

        return loss, orig_loss, dist_loss, q_emb_distill_loss, weight_distillation

    def training_step(self, batch, batch_idx):
        # set to train
        loss, orig_loss, kl_loss, q_emb_distill_loss, weight_distillation = self._step_i(batch, -1)
        self.log("train_loss", loss)
        return {"loss":loss, "orig_loss":orig_loss, "kl_loss":kl_loss,
                "Query_distill_loss":q_emb_distill_loss,
                "Weight_distillation":weight_distillation}
    
    def training_epoch_end(self, outputs):
        # @TODO: 选择在此处输出训练集的准确率
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)

    def validation_step(self, batch, batch_idx):
        result = self.validation_step_i(batch, -1)
        return result
    
    def validation_step_i(self, batch, i):

        inf_result_cache = []
        if self.args.decode_embedding:
            if self.args.position:
                expand_scale = self.args.max_output_length if not self.args.hierarchic_decode else 1
                decode_vocab_size = self.args.output_vocab_size * expand_scale + 2
            else:
                decode_vocab_size = 12
        else:
            decode_vocab_size = None

        assert not self.args.softmax and self.args.gen_method == "greedy"

        if self.args.decode_embedding == 1:
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False, 
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                output_scores=True
            )
            dec = [numerical_decoder(self.args, ids, output=True) for ids in outs]
        elif self.args.decode_embedding == 2:
            if self.args.multiple_decoder:
                target_mask = batch['target_mask'][i].cuda()
            else:
                target_mask = batch['target_mask'].cuda()
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=target_mask,
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,  
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=self.root,
                decoder_index=i,
                output_scores=True
            )
            dec = decode_token(self.args, outs.cpu().numpy())  # num = 10*len(pred)
        else:
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                # no_repeat_ngram_size=2,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,  # False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=self.root,
                output_scores=True
            )
            dec = [self.tokenizer.decode(ids) for ids in outs]

        texts = [self.tokenizer.decode(ids) for ids in batch['source_ids']]

        dec = dec_2d(dec, self.args.num_return_sequences)
        for r in batch['rank']:
            if self.args.label_length_cutoff:
                gt = [s[:self.args.max_output_length - 2] for s in list(r[0])]
            else:
                gt = list(r[0])
            ra = r[1]
            ra = [str(a.item()) for a in ra]

            for pred, g, text, ran in zip(dec, gt, texts, ra):
                pred = ','.join(pred)
                inf_result_cache.append([text, pred, g, int(ran)])
        return {"inf_result_batch": inf_result_cache, 'inf_result_batch_prob': scores}

    def validation_epoch_end(self, outputs):  
        if self.args.multiple_decoder:
            reverse_outputs = []
            for j in range(len(outputs[0])):
                reverse_outputs.append([])
            for i in range(len(outputs)):
                for j in range(len(outputs[0])):
                    reverse_outputs[j].append(outputs[i][j])
            outputs = reverse_outputs

        if self.args.multiple_decoder:
            inf_result_cache = []
            inf_result_cache_prob = []
            for index in range(self.args.decoder_num):
                cur_inf_result_cache = [item for sublist in outputs[index] for item in sublist['inf_result_batch']]
                cur_inf_result_cache_prob = [softmax(sublist['inf_result_batch_prob'][i * int(len(sublist['inf_result_batch_prob'])/len(outputs[index][0]['inf_result_batch'])): (i + 1) * int(len(sublist['inf_result_batch_prob'])/len(outputs[index][0]['inf_result_batch']))]) for sublist in outputs[index] for i in range(len(sublist['inf_result_batch']))]
                inf_result_cache.extend(cur_inf_result_cache)
                inf_result_cache_prob.extend(cur_inf_result_cache_prob)
        else:
            inf_result_cache = [item for sublist in outputs for item in sublist['inf_result_batch']]
            inf_result_cache_prob = [softmax(sublist['inf_result_batch_prob'][i * int(len(sublist['inf_result_batch_prob'])/len(outputs[0]['inf_result_batch'])): (i + 1) * int(len(sublist['inf_result_batch_prob'])/len(outputs[0]['inf_result_batch']))]) for sublist in outputs for i in range(len(sublist['inf_result_batch']))]

        res = pd.DataFrame(inf_result_cache, columns=["query", "pred", "gt", "rank"])
        res.sort_values(by=['query', 'rank'], ascending=True, inplace=True)
        res.to_csv('res.csv', index=False)
        res1 = res.loc[res['rank'] == 1]
        res1 = res1.values.tolist()
        with open("res1.json", 'w') as f:
            json.dump(res1, f)
            

        q_gt, q_pred = {}, {}
        prev_q = ""
        for [query, pred, gt, _] in res1:
            if query != prev_q:
                q_pred[query] = pred.split(",")
                q_pred[query] = q_pred[query][:LAYERS]
                q_pred[query] = list(set(q_pred[query]))
                prev_q = query
            if query in q_gt:
                if len(q_gt[query]) <= 100:
                    q_gt[query].add(gt)
            else:
                q_gt[query] = gt.split(",")
                q_gt[query] = set(q_gt[query])

        total = 0
        for q in q_pred:
            # print(q, q_pred[q], q_gt[q])
            is_hit = 0
            for p in q_gt[q]:
                if p in q_pred[q]:
                    is_hit = 1
            total += is_hit
        recall_avg = total / len(q_pred)
        print("recall@{}:{}".format(self.args.decoder_num, recall_avg))
        self.log("recall1", recall_avg)


    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.decoder_learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.decoder_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.t_total
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    # deleted
    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self): # 调用了
        print('load training data and create training loader.')
        if hasattr(self, 'l1_query_train_dataset'):
            train_dataset = self.l1_query_train_dataset
        else:
            train_dataset = MainDataset(self.args, self.tokenizer) # 如果没有加载则重新加载一遍
        # dataloader中首先获得构造方法里已经准备好的datas
        # print("the_dataset_len:", len(train_dataset));input('s')
        self.prefix_embedding, self.prefix2idx_dict, self.prefix_mask = \
            train_dataset.prefix_embedding, train_dataset.prefix2idx_dict, train_dataset.prefix_mask
        sampler = DistributedSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=self.args.train_batch_size,
                                drop_last=True, shuffle=False, num_workers=4)
        return dataloader


    def val_dataloader(self):
        hues.info('load validation data and create validation loader.')
        val_dataset = MainDataset(self.args, self.tokenizer, path=self.test_file, task='test')
        sampler = DistributedSampler(val_dataset)
        dataloader = DataLoader(val_dataset, sampler=sampler, batch_size=self.args.eval_batch_size,
                                drop_last=True, shuffle=False, num_workers=4)
        return dataloader
