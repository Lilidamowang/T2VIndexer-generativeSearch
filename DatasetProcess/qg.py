import json
import pickle
import glob
import numpy as np
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("/root/autodl-tmp/WebVid/res/vid2emb_msrvtt.pkl", 'rb') as f:
    embeddings_msrvtt = pickle.load(f)

with open("/root/autodl-tmp/WebVid/res/vid2emb_msvd.pkl", 'rb') as f:
    embeddings_t = pickle.load(f)

with open("/root/autodl-tmp/WebVid/res/webvid_rawcaptions.json", 'r') as f:
    raw_captions = json.load(f) 

num = 30
with open(f"/root/autodl-tmp/generateSearch/MSRVTTdataset/kmeans/k{30}_c{30}_seed_34_webvid.pkl", 'rb') as f:
    mapping = pickle.load(f)

from tqdm import tqdm
import json
def find_most_similar_id(target_vector, id_to_vector):
    # target_vector = id_to_vector[target_id]
    most_similar_id = None
    highest_similarity = -1  # 初始化为负数，表示相似度的最小值

    for id, vector in id_to_vector.items():
        similarity = cosine_similarity(target_vector, vector)
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_id = id

    return most_similar_id

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

import numpy as np
def cosine_similarity(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()

    # 计算两个向量的余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity


id2cap = []
for vid in tqdm(embeddings_t):
    embedding = embeddings_t[vid]
    caption = raw_captions[vid]
    most_similar_id = find_most_similar_id(embedding, embeddings_msrvtt)
    tokenseq = mapping[most_similar_id]
    all_cap = []
    for item in caption:
        # all_cap.append(" ".join(item))
        id2cap.append([" ".join(item), most_similar_id, tokenseq])
    # id2cap.append([all_cap, most_similar_id, tokenseq])

with open(f"webvid/msvd_id2cap_{30}.json", 'w') as f:
    json.dump(id2cap, f, cls=NpEncoder)