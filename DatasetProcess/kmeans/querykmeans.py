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
from scipy.spatial import KDTree



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

with open("/root/autodl-tmp/generateSearch/MSRVTTdataset/vid-cap-emb_concat.pkl", 'rb') as f:
    vid_cap_emb = pickle.load(f)


video_ids = []
features = []
for item in vid_cap_emb:
    vid = item[0]
    if vid in train_vids:
        video_ids.append(vid)
        features.append(item[2])
X = np.array(features)
X = X.reshape(X.shape[0], -1)
print(X.shape)


new_id_list = []
kmeans = KMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=args.seed, tol=1e-7)

mini_kmeans = MiniBatchKMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=3,
                              batch_size=1000, reassignment_ratio=0.01, max_no_improvement=20, tol=1e-7)

def classify_recursion(x_data_pos):
    if x_data_pos.shape[0] <= args.c:
        if x_data_pos.shape[0] == 1:
            return
        for idx, pos in enumerate(x_data_pos): 
            new_id_list[pos].append(idx)
        return

    temp_data = np.zeros((x_data_pos.shape[0], 768))
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
for i in range(args.k): 
    print(i, "th cluster")
    pos_lists = [];
    for id_, class_ in enumerate(pred):  
        if class_ == i:  
            pos_lists.append(id_)
    classify_recursion(np.array(pos_lists)) 

# video_features2 = []
# for item in video_features:
#     video_features2.append(item.reshape(512))

# kdtree = KDTree(video_features2)

#print(new_id_list[100:200])
mapping = {}
for i in range(len(video_ids)):
    mapping[video_ids[i]] = new_id_list[i]

for vid in tqdm(val_vids):
    mapping[vid] = [0,0,0,0]

# for vid in tqdm(val_vids):
#     # k = str(vid[5::])
#     target_vector = embeddings_all_videos_msrvtt[vid] # 找到测试集视频的特征
#     nearest_neighbor_idx = kdtree.query(target_vector, k=1)[1] 
#     most_similar_id = video_ids[nearest_neighbor_idx]
#     # most_similar_id = find_most_similar_id(target_vector, embeddings)
#     # tvid = f"video{most_similar_id}"
#     mapping[vid] = mapping[most_similar_id]


with open(f'Query_k{args.k}_c{args.c}_seed_{args.seed}.pkl', 'wb') as f:
    pickle.dump(mapping, f)
