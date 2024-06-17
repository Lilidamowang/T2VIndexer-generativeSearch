from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import argparse
import time
import hues
import torch
from tqdm import tqdm
import os


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--k', type=int, default= 25)   # 30
parser.add_argument('--c', type=int, default= 25)  # 30

args = parser.parse_args()



folder_path = "/root/autodl-nas/git2/GenerativeImage2Text/videocaptioning/frames_ite_1sec"
all_feature = {}
for subfolder_name in tqdm(os.listdir(folder_path)):
    subfolder_path = os.path.join(folder_path, subfolder_name)
    if os.path.isdir(subfolder_path):
        path = [os.path.join(subfolder_path, filename) for filename in os.listdir(subfolder_path) if filename=="temp_feature.pickle"][0]
        with open(path, 'rb') as f:
            feat = pickle.load(f).detach() # [max_len, 512]
            all_feature[subfolder_name] = feat.view(feat.shape[0] * feat.shape[1]).numpy() # [max_len * 512]

video_features = []
video_ids = []
for videoid in all_feature.keys():
    video_ids.append(videoid)
    video_features.append(all_feature[videoid])
# print(train_video_features[0].shape) # train_video_features[0].shape = , 1024
X = np.array(video_features)
print(X.shape) # (10000, 15872)


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

    temp_data = np.zeros((x_data_pos.shape[0], 15872))
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
for i in range(args.k): # 
    print(i, "th cluster")
    pos_lists = [];
    for id_, class_ in enumerate(pred):  
        if class_ == i:  # 
            pos_lists.append(id_)
    classify_recursion(np.array(pos_lists)) # 

#print(new_id_list[100:200])
mapping = {}
for i in range(len(video_ids)):
    mapping[video_ids[i]] = new_id_list[i]

with open(f'k{args.k}_c{args.c}_seed_{args.seed}_CLIPembedding.pkl', 'wb') as f:
    pickle.dump(mapping, f)
