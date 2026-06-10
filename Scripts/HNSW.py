import numpy as np
import pickle

def make_graph(all_embeds,threshold,name):
    graph = {}

    for i,x in enumerate(all_embeds):
        similar = []

        for k,y in enumerate(all_embeds):
            if i == k: continue

            similarity = np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))
            if similarity > threshold:
                similar.append(k)

        graph[i] = similar

    data = []
    with open("../Data/Graph/hnsw_data.pickle","rb") as f:
        data = pickle.load(f)

    if data:
        found = False

        for i,d in enumerate(data):
            if d["name"] == name:
                data[i] = {"name":name,"graph":graph}
                found = True
        if not found:
            data.append({"name": name, "graph": graph})
    else:
        data.append({"name":name,"graph":graph})

    with open("../Data/Graph/hnsw_data.pickle","wb") as f:
        pickle.dump(data,f)

    return graph

def check_graph(query_embed,all_embeds,graph,threshold):
    similar = []
    visited = set()
    pile = [0]

    while pile:
        cur_id = pile.pop()
        if cur_id in visited:
            continue

        cur_e = all_embeds[cur_id]
        visited.add(cur_id)
        sim = np.dot(query_embed,cur_e)/(np.linalg.norm(query_embed)*np.linalg.norm(cur_e))

        if sim>0.96:
            similar.append(cur_id)
            for n in graph[cur_id]:
                if n not in visited:
                    pile.append(n)
    return similar

def compare_embed(all_embeds,query_embed,name,threshold):
    data = []
    graph = {}

    with open("../Data/Graph/hnsw_data.pickle","rb") as f:
        data = pickle.load(f)

    for d in data:
        if d["name"] == name:
            graph = d["graph"]

    similar = check_graph(query_embed,all_embeds,graph,threshold)

    return similar