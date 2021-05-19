"""
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao

Modified on Nov 15, 2019
@author: Shuxun Zan
"""
import random
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super(AGREE, self).__init__()
        self.userembeds = nn.Embedding(num_users, embedding_dim)
        self.itemembeds = nn.Embedding(num_items, embedding_dim)
        self.groupembeds = nn.Embedding(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        random_gid = random.sample(list(self.group_member_dict.keys()), 1)
        self.group_size = len(self.group_member_dict[random_gid[0]])
        print('self.group_size', self.group_size)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, user_inputs, item_inputs, group_members):
        # train group
        if (group_inputs is not None) and (user_inputs is None):
            out = self.grp_forward(group_inputs, item_inputs, group_members)
        # train user
        else:
            out = self.usr_forward(user_inputs, item_inputs)
        return out

    # group forward
    def grp_forward(self, group_inputs, item_inputs, group_members):
        item_embeds_full = self.itemembeds(item_inputs)
        members_embeds = self.userembeds(group_members)
        item_embeds = self.itemembeds(item_inputs)

        at_wt = self.attention(members_embeds, item_embeds)
        at_wt = at_wt.reshape(at_wt.shape[0], 1, self.group_size)
        g_embeds_with_attention = torch.bmm(at_wt, members_embeds)
        g_embeds_with_attention = torch.squeeze(g_embeds_with_attention)

        group_embeds_pure = self.groupembeds(group_inputs)
        g_embeds = g_embeds_with_attention + group_embeds_pure

        element_embeds = torch.mul(g_embeds, item_embeds_full)  # Element-wise product
        new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

    # user forward
    def usr_forward(self, user_inputs, item_inputs):
        user_embeds = self.userembeds(user_inputs)
        item_embeds = self.itemembeds(item_inputs)
        element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, members_embeds, item_embeds):
        outputs = []
        for j in range(members_embeds.shape[1]):
            u = members_embeds[:, j]
            u_j = torch.cat((u, item_embeds), 1)
            outputs.append(self.linear(u_j))
        output_cat = torch.cat(outputs, dim=1)
        weights = torch.softmax(output_cat, dim=1)
        return weights


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
