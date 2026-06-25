import pickle
import numpy as np
from loader import console

path_graph = "../Data/Graph/hnsw_data.pickle"

def load_data():
    try:
        with open(path_graph, "rb") as f:
            return pickle.load(f)

    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return []

def save_data(data):
    with open(path_graph, "wb") as f:
        pickle.dump(data, f)

def make_graph(all_embeds,name,save=True):
    all_embeds = np.array(all_embeds, dtype=np.float32)
    norms = np.linalg.norm(all_embeds, axis=1, keepdims=True)
    all_embeds_norm = all_embeds / np.clip(norms, 1e-8, None)

    graph = []
    k = min(8, len(all_embeds_norm) - 1)

    for i,x in enumerate(all_embeds_norm):
        sim = all_embeds_norm @ x
        sim[i] = -np.inf
        sim_ids = np.argpartition(sim, -k)[-k:]

        graph.append(sim_ids.tolist())

    mean = np.mean(all_embeds_norm, axis=0)
    mean_norm = np.linalg.norm(mean)

    if mean_norm > 1e-8:
        mean /= mean_norm

    center_node = int(np.argmax(all_embeds_norm @ mean))

    if save:
        data = load_data()
        data.append({"name":name,"graph":graph,"center_node":center_node})
        save_data(data)

    return graph

def add_to_graph(name,all_embeds,query_embed):
    data = load_data()

    center = 0
    graph = []

    for d in data:
        if d["name"] == name:
            center = d["center_node"]
            graph = d["graph"]
            break

    all_embeds = np.array(all_embeds, dtype=np.float32)
    norms = np.linalg.norm(all_embeds, axis=1, keepdims=True)
    all_embeds_norm = all_embeds / np.clip(norms, 1e-8, None)
    query_norm = query_embed/np.linalg.norm(query_embed)

    threshold = -np.inf

    visited = set()
    similar = set()
    stack = list()

    stack.append(center)

    off_set_allowed = 0.15

    while stack:
        cur_id = stack.pop()

        if cur_id in visited:
            continue

        visited.add(cur_id)

        sim = all_embeds_norm[cur_id] @ query_norm

        if sim > threshold:
            threshold = sim
            similar.clear()
            similar.add(int(cur_id))

        elif sim >= threshold - off_set_allowed:
            similar.add(int(cur_id))

        stack.extend(graph[cur_id])

    graph.append(list(similar)[:min(8,len(similar)-1)])

    for s in similar:
        graph[s].append(len(graph)-1)

    data = load_data()

    for d in data:
        if d["name"] == name:
            d["graph"] = graph
            break

    save_data(data)

def check_graph(all_embeds, query_embed, graph, threshold, max_search=5,max_depth=10):
    center_node = graph["center_node"]

    query_norm = query_embed/np.linalg.norm(query_embed)
    all_embeds_norm = all_embeds/np.linalg.norm(all_embeds,axis=1,keepdims=True)

    visited = set()
    similar = set()
    stack = list()

    number_of_max_searches = 0

    depth = 0
    stack.append((center_node,depth))

    while stack and number_of_max_searches < max_search:
        cur_id,d = stack.pop()

        if cur_id in visited or d >= max_depth :
            continue

        number_of_max_searches += 1

        visited.add(cur_id)

        sim = all_embeds_norm[cur_id] @ query_norm
        sim = sim.flatten()
        console.print(f"[dim]sim centre: {sim}[/dim]")

        if sim > threshold:
            similar.add(int(cur_id))

            for s in graph[cur_id]:
                if s not in visited:
                    stack.append((s,depth+1))

    return similar

def compare_embed(all_embeds, query_embed, name, threshold,depth):
    data = load_data()
    graph = {}

    for idx, d in enumerate(data):
        if d["name"] == name:
            graph = d["graph"]

    return check_graph(all_embeds, query_embed, graph, threshold,depth)