import os
import json
import pandas as pd
import pickle
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import trange, tqdm
import random
num = 50
info = "_only_qg"
os.system(f'mkdir /root/autodl-nas/generateSearch/MSRVTTdataset/k{num}_c{num}_CLIP{info}')
path = f"kmeans/k{num}_c{num}_seed_7_CLIPembedding.pkl"
ex_train = f"/root/autodl-nas/generateSearch/MSRVTTdataset/k{num}_c{num}_CLIP{info}/train.tsv"
ex_val = f"/root/autodl-nas/generateSearch/MSRVTTdataset/k{num}_c{num}_CLIP{info}/test.tsv"

with open(path, 'rb') as f:
    kmeans_dict = pickle.load(f)
new_kmeans_dict = {}
for old_docid in kmeans_dict.keys():
    new_kmeans_dict[str(old_docid)] = '-'.join(str(elem) for elem in kmeans_dict[old_docid])

with open("../data/MSRVTT/train_list_jsfusion.txt", "r") as f:
    train_list_id = f.readlines()
for i, item in enumerate(train_list_id):
    if '\n' in item:
        item = item.replace('\n', '')
    train_list_id[i] = item
    
with open("../data/MSRVTT/annotation/MSR_VTT.json", "r") as f:
    msr_vtt = json.load(f)
annotations = msr_vtt['annotations']

with open("../data/MSRVTT/val_list_jsfusion.txt", "r") as f:
    val_list_id = f.readlines()
for i, item in enumerate(val_list_id):
    if '\n' in item:
        item = item.replace('\n', '')
    val_list_id[i] = item
    
with open("../data/MSRVTT/raw-captions.pkl", "rb") as f:
    raw_captions = pickle.load(f)
with open("../data/MSRVTT/structured-symlinks/jsfusion_val_caption_idx.pkl", "rb") as f:
    jsfusion_val_caption_idx = pickle.load(f)
    
with open("/root/autodl-nas/git2/GenerativeImage2Text/videocaptioning/results_20.json", 'r') as f:
    qg = json.load(f)
    
"""
原始训练集
"""

count_train_anno = 0
count_val_anno = 0

file_train = open(ex_train, 'w')
for videoid in tqdm(new_kmeans_dict.keys()):
    if videoid in train_list_id:
        kmeans_id = new_kmeans_dict[videoid]
        for item in annotations:
            if item['image_id'] == videoid:
                caption = item['caption']
                file_train.write('\t'.join([caption, videoid, kmeans_id]) + '\n')
                file_train.flush()
                count_train_anno += 1
# file_train = open(ex_train, 'w')
# for videoid in tqdm(new_kmeans_dict.keys()):
#     if videoid in train_list_id:
#         flg = True
#     else:
#         flg = False
#     kmeans_id = new_kmeans_dict[videoid]
#     for item in annotations:
#         if flg:
#             if item['image_id'] == videoid:
#                 caption = item['caption']
#                 file_train.write('\t'.join([caption, videoid, kmeans_id]) + '\n')
#                 file_train.flush()
#                 count_train_anno += 1
#         else: # 在测试集中
#             if item['image_id'] == videoid:
#                 caption = item['caption']
#                 # 当前caption与测试caption不符 则加入到测试数据
#                 if caption != " ".join(raw_captions[videoid][int(jsfusion_val_caption_idx[videoid])]):
#                     file_train.write('\t'.join([caption, videoid, kmeans_id]) + '\n')
#                     file_train.flush()
#                     count_val_anno += 1
print(count_train_anno)
print(count_val_anno)

"""
数据集扩充：qg
"""
count_qg = 0
for videoid in tqdm(qg.keys()):
    videoid_k = videoid[0: videoid.rfind('.')]
    kmeans_id = new_kmeans_dict[videoid_k]
    for caption in qg[videoid]:
        file_train.write('\t'.join([caption, videoid_k, kmeans_id]) + '\n')
        file_train.flush()
        count_qg += 1
print(count_qg)

"""

"""
file_train = open(ex_val, 'w')

for videoid in tqdm(val_list_id):
    kmeans_id = new_kmeans_dict[videoid]
    caption = " ".join(raw_captions[videoid][int(jsfusion_val_caption_idx[videoid])])
    file_train.write('\t'.join([caption, videoid, kmeans_id]) + '\n')
    file_train.flush()
