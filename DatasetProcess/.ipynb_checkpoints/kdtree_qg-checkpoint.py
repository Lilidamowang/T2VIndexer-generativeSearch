import json
import pickle
import glob
import numpy as np
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial import KDTree

with open("/root/autodl-tmp/WebVid/res/vid2emb_msrvtt.pkl", 'rb') as f:
    embeddings_msrvtt = pickle.load(f)

with open("/root/autodl-tmp/WebVid/res/vid2emb_webvid.pkl", 'rb') as f:
    embeddings_t = pickle.load(f)

# with open("/root/autodl-tmp/generateSearch/data/MSVD/msvd_data/raw-captions.pkl", 'rb') as f:
#     raw_captions = pickle.load(f) 

with open("/root/autodl-tmp/WebVid/res/webvid_rawcaptions.json", 'r') as f:
    raw_captions = json.load(f)

num = 30
with open(f"/root/autodl-tmp/generateSearch/MSRVTTdataset/kmeans/k{30}_c{30}_seed_34_webvid.pkl", 'rb') as f:
    mapping = pickle.load(f)

t_vid = []
t_emb = []
msrvtt_vid = []
msrvtt_emb = []
for k in tqdm(embeddings_t):
    t_vid.append(k)
    t_emb.append(embeddings_t[k])
for k in tqdm(embeddings_msrvtt):
    msrvtt_vid.append(k)
    msrvtt_emb.append(embeddings_msrvtt[k])

# 构建 KD 树
kdtree = KDTree(msrvtt_emb)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


id2cap = []
for vid in tqdm(embeddings_t):
    caption = raw_captions[vid]
    
    # 查找最近邻的向量索引
    nearest_neighbor_idx = kdtree.query(embeddings_t[vid], k=1)[1]
    msrvid = msrvtt_vid[nearest_neighbor_idx]
    tokenseq = mapping[msrvid]
    #for item in caption:
        # id2cap.append([" ".join(item), msrvid, tokenseq])
    id2cap.append([caption, msrvid, tokenseq])
    if len(id2cap)==7000000:
        with open(f"webvid/webvid_id2cap_{30}_7M.json", 'w') as f:
            json.dump(id2cap, f, cls=NpEncoder)
    elif len(id2cap)==1000000:
        with open(f"webvid/webvid_id2cap_{30}_1M.json", 'w') as f:
            json.dump(id2cap, f, cls=NpEncoder)
    elif len(id2cap)==5000000:
        with open(f"webvid/webvid_id2cap_{30}_5M.json", 'w') as f:
            json.dump(id2cap, f, cls=NpEncoder)

with open(f"webvid/webvid_id2cap_{30}_10M.json", 'w') as f:
    json.dump(id2cap, f, cls=NpEncoder)