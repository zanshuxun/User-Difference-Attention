"""
Created on Nov 10, 2017
Deal something

@author: Lianhai Miao

Modified on Nov 15, 2019
@author: Shuxun Zan
"""
import torch
import numpy as np
import math
import heapq
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Helper(object):
    """
        utils class: it can provide any function that we need
    """

    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line is not None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        self.g_m_d = g_m_d
        return g_m_d

    def evaluate_model(self, model, testRatings, testNegatives, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        for _ in [5, 10, 15, 20]:
            hits.append([])
            ndcgs.append([])

        for idx in tqdm(range(len(testRatings))):
            (hr, ndcg) = self.eval_one_rating(model, testRatings, testNegatives, [5, 10, 15, 20], type_m, idx)
            for i, _ in enumerate(hr):
                hits[i].append(hr[i])
                ndcgs[i].append(ndcg[i])
        return (hits, ndcgs)

    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        # items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.LongTensor(users).to(device)
        items_var = torch.LongTensor(items).to(device)

        if type_m == 'group':
            group_members = [self.g_m_d[u]] * len(items)
            group_members = torch.LongTensor(group_members).to(device)
            predictions = model(users_var, None, items_var, group_members)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var, None)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.cpu().numpy()[i]
        # items.pop()
        # Evaluate top rank list
        hrs, ndcgs = [], []
        for top_k in K:
            ranklist = heapq.nlargest(top_k, map_item_score, key=map_item_score.get)
            hr = self.getHitRatio(ranklist, gtItem)
            ndcg = self.getNDCG(ranklist, gtItem)
            hrs.append(hr)
            ndcgs.append(ndcg)
        return (hrs, ndcgs)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i + 2)
        return 0
