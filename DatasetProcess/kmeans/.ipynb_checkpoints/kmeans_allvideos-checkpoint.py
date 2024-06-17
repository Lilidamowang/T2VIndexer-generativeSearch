from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import argparse
import hues
import torch
from tqdm import tqdm
import os

"""
1. 针对MSR_VTT集的train及DiDeMo、MSVD的所有视频进行聚类
2. 对于MSRVTT test的视频，根据表征相似性，选择最相似的视频id序列
"""

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=34)
parser.add_argument('--k', type=int, default= 30)   # 30
parser.add_argument('--c', type=int, default= 30)  # 30

args = parser.parse_args()

with open("/root/autodl-tmp/generateSearch/data/MSRVTT/train_list_jsfusion.txt", 'r') as f:
    lines = f.readlines()
    train_vids = [line.strip() for line in lines]
with open("/root/autodl-tmp/generateSearch/data/MSRVTT/val_list_jsfusion.txt", 'r') as f:
    lines = f.readlines()
    val_vids = [line.strip() for line in lines]

with open("/root/autodl-tmp/CLIP4Clip/res/embeddings_all_videos.pkl",'rb') as f:
    embeddings_all_videos = pickle.load(f) # 10000 个视频的特征 (1,512)
with open("/root/autodl-tmp/CLIP4Clip/res/MSVD/embeddings_all_videos.pkl")
    embeddings_all_videos_msvd = pickle.load(f)
with open("/root/autodl-tmp/CLIP4Clip/res/DiDeMo/embeddings_all_videos.pkl")
    embeddings_all_videos_didemo = pickle.load(f)
    
video_ids = []
video_features = []
for k in embeddings_all_videos:
    vid = f"video{k}"
    if vid in train_vids:
        video_ids.append(vid)
        video_features.append(embeddings_all_videos[k])
for k in embeddings_all_videos_didemo:
    vid = f"didemo_{k}"
    video_ids.append(vid)
    video_features.append(embeddings_all_videos_didemo[k])
for k in embeddings_all_videos_msvd:
    vid = f"msvd_{k}"
    video_ids.append(vid)
    video_features.append(embeddings_all_videos_msvd[k])
X = np.array(video_features)
X = X.reshape(X.shape[0], -1)
print(X.shape)

# KMeans分层聚类
new_id_list = []
kmeans = KMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=args.seed, tol=1e-7)

mini_kmeans = MiniBatchKMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=3,
                              batch_size=1000, reassignment_ratio=0.01, max_no_improvement=20, tol=1e-7)

def classify_recursion(x_data_pos):
    if x_data_pos.shape[0] <= args.c:
        if x_data_pos.shape[0] == 1:
            return
        for idx, pos in enumerate(x_data_pos): # 对第pos个item添加新的id后缀
            new_id_list[pos].append(idx)
        return

    temp_data = np.zeros((x_data_pos.shape[0], 512))
    for idx, pos in enumerate(x_data_pos):
        temp_data[idx, :] = X[pos]

    if x_data_pos.shape[0] >= 1e3:
        pred = mini_kmeans.fit_predict(temp_data)
    else:
        pred = kmeans.fit_predict(temp_data)

    for i in range(args.k):
        pos_lists = []
        for id_, class_ in enumerate(pred):
            if class_ == i:
                pos_lists.append(x_data_pos[id_])
                new_id_list[x_data_pos[id_]].append(i)
        classify_recursion(np.array(pos_lists)) 

    return

print('Start First Clustering')
pred = mini_kmeans.fit_predict(X)
print(pred.shape)   #int 0-k for each vector (10000,)
print(mini_kmeans.n_iter_)

for class_ in pred:
    new_id_list.append([class_])

print('Start Recursively Clustering...')
for i in range(args.k): # 再对第0堆
    print(i, "th cluster")
    pos_lists = [];
    for id_, class_ in enumerate(pred):  
        if class_ == i:  # 所有的第0堆的item
            pos_lists.append(id_)
    classify_recursion(np.array(pos_lists)) # 传入所有属于第0堆的item [5, 12 ,14 ,15 ...], 对这些堆进行聚类

def cosine_similarity(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()

    # 计算两个向量的余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def find_most_similar_id(target_id, id_to_vector):
    target_vector = id_to_vector[target_id]
    most_similar_id = None
    highest_similarity = -1  # 初始化为负数，表示相似度的最小值

    for id, vector in id_to_vector.items():
        if f"video{id}" in val_vids:
            continue  # 跳过自己和测试集
        similarity = cosine_similarity(target_vector, vector)
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_id = id

    return most_similar_id

#print(new_id_list[100:200])
mapping = {}
for i in range(len(video_ids)):
    mapping[video_ids[i]] = new_id_list[i]

# @TODO: 此处改为和所有的视频计算相似性，而不是仅有msrvtt中的视频
for vid in tqdm(val_vids):
    k = str(vid[5::])
    most_similar_id = find_most_similar_id(k, embeddings_all_videos)
    tvid = f"video{most_similar_id}"
    mapping[vid] = mapping[tvid]


with open(f'k{args.k}_c{args.c}_seed_{args.seed}_allvideo.pkl', 'wb') as f:
    pickle.dump(mapping, f)
