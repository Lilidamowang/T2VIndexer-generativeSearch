from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import argparse
import hues
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--k', type=int, default= 30)   # 30
parser.add_argument('--c', type=int, default= 30)  # 30

args = parser.parse_args()

# # 获得训练用的video id及对应的s3d feature
# train_video_ids_path = "/root/autodl-nas/generateSearch/data/MSRVTT/train_list_jsfusion.txt"
# train_video_ids = []
# with open(train_video_ids_path, 'r') as f:
#     train_video_ids = f.readlines()
s3d_feature_path = "/root/autodl-nas/generateSearch/data/MSRVTT/mmt_feats/features.s3d.pkl"
with open(s3d_feature_path, 'rb') as f:
    s3d_features = pickle.load(f)
video_features = []
video_ids = []
for videoid in s3d_features.keys():
    video_ids.append(videoid)
    video_features.append(s3d_features[videoid].mean(0))
# print(train_video_features[0].shape) # train_video_features[0].shape = , 1024
X = np.array(video_features)
# print(X.shape);input('s') # (10000, 1024)

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

    temp_data = np.zeros((x_data_pos.shape[0], 1024))
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
    classify_recursion(np.array(pos_lists)) # 传入所有属于第0堆的item [5, 12 ,14 ,15 ...]

#print(new_id_list[100:200])
mapping = {}
for i in range(len(video_ids)):
    mapping[video_ids[i]] = new_id_list[i]

with open(f'k{args.k}_c{args.c}_seed_{args.seed}.pkl', 'wb') as f:
    pickle.dump(mapping, f)