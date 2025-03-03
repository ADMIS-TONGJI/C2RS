import numpy as np
import dgl
from dgl.nn import RelGraphConv

def calculate_rank(score, target, filter_list):
    score_target = score[target]
    score[filter_list] = score_target - 1
    rank = np.sum(score > score_target) + np.sum(score == score_target) // 2 + 1
    score[target] += 1
    return rank

def get_topk_indices(score, target, filter_list, topk):
    score_target = score[target]
    score[filter_list] = score_target - 1
    score[target] += 1
    sorted_indices = np.argsort(score)[-topk:][::-1]
    return sorted_indices

def metrics(rank):
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1

def build_kg(dataset, num_ent, num_rel):
    path = "./data/{}/train.txt".format(dataset)
    f = open(path, 'r')
    e1, e2, rels = [], [], []
    entity_map = {}
    for line in f.readlines()[1:]:
        h, t, r = line[:-1].split('\t')
        e1.append(int(h))
        e2.append(int(t))
        rels.append(int(r))
        entity_map[int(h)] = 1
        entity_map[int(t)] = 1
    for i in range(0, num_ent):
        e1.append(i)
        e2.append(i)
        rels.append(num_rel)
    graph = dgl.graph((e1, e2))
    return graph, rels