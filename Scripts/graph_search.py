import pickle
import numpy as np
from loader import console

path_graph = "../Data/Graph/graph_data.pickle"

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

    if len(all_embeds_norm) == 1:
        graph = [[]]
        center_node = 0

        if save:
            data = load_data()
            data.append({"name": name,"graph": graph,"center_node": center_node,"all_embeds": all_embeds_norm})
            save_data(data)

        return graph, center_node, all_embeds_norm

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
        data.append({"name":name,"graph":graph,"center_node":center_node,"all_embeds":all_embeds_norm})
        save_data(data)

    return graph,center_node,all_embeds_norm

def add_to_graph(all_embeds,query_embed,graph,center):
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

    graph.append(list(similar)[:min(8, len(similar))])

    for s in similar:
        graph[s].append(len(graph)-1)

    all = np.vstack([all_embeds_norm,query_norm])
    mean = np.mean(all, axis=0)
    mean_norm = np.linalg.norm(mean)
    mean = mean/mean_norm

    center = int(np.argmax(all @ mean))

    return graph,center,np.vstack([all_embeds_norm,query_norm])

def update_graph(name, query_embed):
    data = load_data()
    graph = []
    center = None
    all_embeds = []

    for d in data:
        if d["name"] == name:
            graph = d["graph"]
            center = d["center_node"]
            all_embeds = d["all_embeds"]

    if query_embed.ndim == 1:
        graph,center,all_embeds = add_to_graph(all_embeds,query_embed,graph,center)
    else:
        for q in query_embed:
            graph,center,all_embeds = add_to_graph(all_embeds,q,graph,center)

    for d in data:
        if d["name"] == name:
            d["graph"] = graph
            d["center_node"] = center
            d['all_embeds'] = all_embeds

    save_data(data)

def check_graph(all_embeds, query_embed, graph, max_search=5,max_depth=10,center_node=None):
    q_norm = np.linalg.norm(query_embed)
    query_norm = query_embed / np.clip(q_norm, 1e-8, None)

    visited = set()
    similar = set()
    stack = list()

    number_of_max_searches = 0

    depth = 0
    stack.append((center_node,depth))

    threshold = -np.inf
    offset = 0.2

    while stack and number_of_max_searches < max_search:
        cur_id,d = stack.pop()

        if cur_id in visited or d >= max_depth :
            continue

        number_of_max_searches += 1

        visited.add(cur_id)

        sim = all_embeds[cur_id] @ query_norm

        console.print(f"[dim]sim centre: {sim}[/dim]")

        if sim > threshold:
            threshold = sim
            similar.clear()
            similar.add(int(cur_id))

        elif sim >= threshold - offset:
            similar.add(int(cur_id))

        for n in graph[cur_id]:
            stack.append((n, d + 1))

    return similar

def compare_embed(query_embed, name,depth):
    data = load_data()
    graph = {}
    center = None
    all_embeds = []

    for idx, d in enumerate(data):
        if d["name"] == name:
            graph = d["graph"]
            center = d["center_node"]
            all_embeds = d["all_embeds"]

    return check_graph(all_embeds=all_embeds, query_embed=query_embed, graph=graph,max_depth=depth,center_node=center)