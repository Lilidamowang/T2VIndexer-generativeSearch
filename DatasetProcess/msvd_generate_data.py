import os
import json
import pandas as pd
import pickle
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import trange, tqdm
import random
num = 30
info="webvid_c4v_qg_layern"
model = "D"
WTC = 0
if not os.path.exists("/root/autodl-tmp/generateSearch/MSRVTTdataset/k{num}_c{num}_{info}"):
    os.system(f'mkdir /root/autodl-tmp/generateSearch/MSRVTTdataset/k{num}_c{num}_{info}')
path = f"kmeans/k{num}_c{num}_seed_34_webvid2.pkl"
# /root/autodl-tmp/generateSearch/MSRVTTdataset/kmeans/k30_c30_seed_34_webvid2.pkl
ex_train = f"/root/autodl-tmp/generateSearch/MSRVTTdataset/k{num}_c{num}_{info}/train.tsv"
ex_val = f"/root/autodl-tmp/generateSearch/MSRVTTdataset/k{num}_c{num}_{info}/test.tsv"

with open(path, 'rb') as f:
    kmeans_dict = pickle.load(f)
new_kmeans_dict = {}
for old_docid in kmeans_dict.keys():
    ts = kmeans_dict[old_docid]
    if len(ts) > 2:
        ts = ts[0:-2]
    elif len(ts) > 1:
        ts = ts[0:-1]
    # ts = [str(ts[0])]
    new_kmeans_dict[str(old_docid)] = '-'.join(str(elem) for elem in ts)
    # new_kmeans_dict[str(old_docid)] = str(ts[0])

with open("../data/MSRVTT/train_list_jsfusion.txt·", "r") as f:
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

with open("/root/autodl-tmp/mPLUG-Owl/new_v2c.json", 'r') as f:
    v2c = json.load(f)

with open("msrvtt_video2caption.json", 'r') as f:
    msrvtt_video2caption = json.load(f)
with open("/root/autodl-tmp/LocalizingMoments/vid2caps.json", 'r') as f:
    didemo = json.load(f)
didemo_video2caption = {}
for k in didemo:
    didemo_video2caption[k.split('.')[0]] = didemo[k]
with open("/root/autodl-tmp/generateSearch/data/MSVD/msvd_data/raw-captions.pkl", 'rb') as f:
    msvd_video2caption = pickle.load(f) 

"""
原始训练集
"""
count_train_anno = 0
count_val_anno = 0
print("model D")
file_train = open(ex_train, 'w')
for videoid in tqdm(new_kmeans_dict.keys()):
    kmeans_id = new_kmeans_dict[videoid]
    if msrvtt_video2caption[videoid][0][1]: # 如果是训练集
        for caption in msrvtt_video2caption[videoid]:
            file_train.write('\t'.join([caption[0], videoid, kmeans_id]) + '\n')
            file_train.flush()
            count_train_anno += 1
#             for caption in v2c[videoid]:
#                 file_train.write('\t'.join([caption, videoid, kmeans_id]) + '\n')
#                 file_train.flush()
#                 count_train_anno += 1
    else:
        for i in range(WTC):
            caption = msrvtt_video2caption[videoid][i][0]
            file_train.write('\t'.join([caption, videoid, kmeans_id]) + '\n')
            file_train.flush()
            count_train_anno += 1
#         # 加入qg项
    for caption in v2c[videoid]:
        file_train.write('\t'.join([caption, videoid, kmeans_id]) + '\n')
        file_train.flush()
        count_train_anno += 1
# 加入cap4video
with open("/root/autodl-tmp/generateSearch/MSRVTTdataset/msrvtt_cap4video_id2cap_10.json", 'r') as f:
    cap4video_id2cap = json.load(f)
for item in tqdm(cap4video_id2cap):
#         if item[1] in val_list_id:
#             continue
    elems = item[2]
    if len(elems) > 2:
        elems = elems[0:-2]
    elif len(elems) > 1:
        elems = elems[0:-1]
    # elems = [str(elems[0])]
    file_train.write('\t'.join([item[0], item[1], '-'.join(str(elem) for elem in elems)]) + '\n')
    # file_train.write('\t'.join([item[0], item[1], str(elems[0])]) + '\n')
    file_train.flush()
    count_train_anno += 1
# 加入msvd项
#     with open(f"webvid/msvd_id2cap_{num}.json", 'r') as f:
#         msvd_id2cap = json.load(f)
#     for item in tqdm(msvd_id2cap):
#         file_train.write('\t'.join([item[0], item[1], '-'.join(str(elem) for elem in item[2])]) + '\n')
#         file_train.flush()
#         count_train_anno += 1
# # '-'.join(str(elem) for elem in kmeans_dict[old_docid])
# 加入DiDeMo项
#     with open(f"webvid/didemo_id2cap_{num}.json", 'r') as f:
#         didemo_id2cap = json.load(f)
#     for item in didemo_id2cap:
#         file_train.write('\t'.join([item[0], item[1], '-'.join(str(elem) for elem in item[2])]) + '\n')
#         file_train.flush()
#         count_train_anno += 1
# 加入webvid项：
#     with open(f"webvid/webvid_id2cap_{num}_1M.json", 'r') as f:
#         webvid_id2cap = json.load(f)
#     for item in tqdm(webvid_id2cap):
#         file_train.write('\t'.join([item[0].replace('\t', '').replace('\n', ''), item[1], '-'.join(str(elem) for elem in item[2])]) + '\n')
#         file_train.flush()
#         count_train_anno += 1
print(count_train_anno)
"""
测试集
"""
file_val = open(ex_val, 'w')

for videoid in tqdm(val_list_id):
    kmeans_id = new_kmeans_dict[videoid]
    caption = " ".join(raw_captions[videoid][int(jsfusion_val_caption_idx[videoid])])
    file_val.write('\t'.join([caption, videoid, kmeans_id]) + '\n')
    file_val.flush()
