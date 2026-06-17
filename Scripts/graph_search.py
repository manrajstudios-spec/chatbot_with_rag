import heapq
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

def make_graph(all_embeds, name, save):
    if all_embeds.ndim == 1:
        all_embeds = all_embeds[None, :]

    all_embeds = np.array(all_embeds, dtype=np.float32)
    norms = np.linalg.norm(all_embeds, axis=1, keepdims=True)
    norm_all = all_embeds / np.clip(norms, 1e-8, None)

    graph = {}

    for i, x in enumerate(norm_all):
        sim = norm_all @ x
        sim[i] = -np.inf

        k = min(8, len(sim) - 1)

        neighbors = np.argpartition(sim, -k)[-k:]
        neighbors = neighbors[np.argsort(sim[neighbors])[::-1]]
        graph[i] = neighbors.tolist()

    mean = np.mean(norm_all, axis=0)
    mean_norm = np.linalg.norm(mean)

    if mean_norm > 1e-8:
        mean /= mean_norm

    center_node = int(np.argmax(norm_all @ mean))

    if save:
        data = load_data()

        for idx, d in enumerate(data):
            if d["name"] == name:
                data[idx] = {"name": name,"graph": graph,"center_node": center_node}
                break
        else:
            data.append({"name": name,"graph": graph,"center_node": center_node})

        save_data(data)

    return graph, center_node
def check_graph(query_embed, all_embeds, graph, threshold, start):
    similar = []
    visited = set()

    norm_query = query_embed / np.linalg.norm(query_embed)
    norm_all = all_embeds / np.linalg.norm(all_embeds,axis=1,keepdims=True)
    start_sim = float(norm_query @ norm_all[start])

    heap = [(-start_sim, start)]
    heapq.heapify(heap)

    while heap:
        n_similarity, cur_id = heapq.heappop(heap)
        if cur_id in visited:
            continue
        visited.add(cur_id)

        console.print(f"[dim]sim centre: {n_similarity}[/dim]")

        sim = -n_similarity

        if sim > threshold:
            similar.append(cur_id)

        for neighbor in graph[cur_id]:
            if neighbor not in visited:
                neighbor_sim = float(norm_query @ norm_all[neighbor])
                console.print(f"[dim]neighbour: {neighbor_sim}[/dim]")
                heapq.heappush(heap,(-neighbor_sim, neighbor))

    return similar
def compare_embed(all_embeds, query_embed, name, threshold):
    data = load_data()
    start = 0
    graph = {}

    for idx, d in enumerate(data):
        if d["name"] == name:
            graph = d["graph"]
            start = d["center_node"]

    if not graph:
        graph = make_graph(all_embeds, threshold, name, save=True)

    return check_graph(query_embed, all_embeds, graph, threshold,start)