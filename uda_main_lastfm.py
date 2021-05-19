"""
Created on Nov 10, 2017
Main function

@author: Lianhai Miao

Modified on Nov 15, 2019
@author: Shuxun Zan
"""
import torch
import argparse
import numpy as np

from torch import optim
from time import time
from tqdm import tqdm

from model.uda import UDA
from util import Helper
from dataset import GDataset

# torch.cuda.set_device(3)  # specify a gpu device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Run UDA.")
    parser.add_argument('--dataset', nargs='?', default='ml100k_1000_3',
                        help='dataset')
    parser.add_argument('--embedding_size', type=int, default='32',
                        help='embedding_size')
    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--num_negatives', type=int, default=5,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='')
    parser.add_argument('--lr', nargs='?', default='[0.01, 0.005, 0.001]',
                        help="lr.")
    return parser.parse_args()


# train the model
def training(model, train_loader, epoch_id, args, type_m):
    # user training
    learning_rates = args.lr

    # learning rate decay
    lr = learning_rates[0]
    if 15 <= epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >= 20:
        lr = learning_rates[2]
    # lr decay
    if epoch_id % 5 == 0:
        lr /= 2

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)

    # losses = []
    print(len(train_loader))
    for batch_id, (u, group_members, pi_ni) in tqdm(enumerate(train_loader)):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]
        # Forward
        if type_m == 'user':
            pos_prediction = model(None, user_input, pos_item_input, None)
            neg_prediction = model(None, user_input, neg_item_input, None)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input, group_members)
            neg_prediction = model(user_input, None, neg_item_input, group_members)
        # Zero_grad
        model.zero_grad()
        # Loss
        # print(pos_prediction,'\n',neg_prediction)
        loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
        # record loss history
        # losses.append(loss)
        # Backward
        loss.backward()
        optimizer.step()

    print('Iteration %d' % epoch_id)
    loss = loss.data.cpu().numpy()
    print('loss', loss)

    return loss


def evaluation(model, helper, testRatings, testNegatives, type_m):
    model.eval()
    (hits, ndcgs) = helper.evaluate_model(
        model, testRatings, testNegatives, type_m)

    mean_hrs, mean_ndcgs = [], []
    for i, _ in enumerate(hits):
        hr, ndcg = np.array(hits[i]).mean(), np.array(ndcgs[i]).mean()
        mean_hrs.append(hr)
        mean_ndcgs.append(ndcg)
    return mean_hrs, mean_ndcgs


if __name__ == '__main__':
    args = parse_args()
    args.lr = eval(args.lr)

    args.path = './data/' + args.dataset + '/'
    args.user_dataset = args.path + 'userRating'
    args.group_dataset = args.path + 'groupRating'
    args.user_in_group_path = "./data/" + args.dataset + "/groupMember.txt"

    print('args', args)
    # initial helper
    helper = Helper()

    # get the dict of users in group
    g_m_d = helper.gen_group_member_dict(args.user_in_group_path)

    # initial dataSet class
    dataset = GDataset(args.user_dataset,
                       args.group_dataset, args.num_negatives, g_m_d)

    # get group number
    num_group = max(g_m_d.keys()) + 1
    num_users, num_items = dataset.num_users, dataset.num_items
    # print('num_users',num_users)
    # build UDA model
    agree = UDA(num_users, num_items, num_group,
                  args.embedding_size, g_m_d, args.drop_ratio).to(device)
    # config information
    print("UDA at embedding size %d, run Iteration:%d" %
          (args.embedding_size, args.epoch))
    history = []
    tops = [5, 10, 15, 20]
    for epoch in range(args.epoch):
        # training
        agree.train()
        t1 = time()
        training(agree, dataset.get_user_dataloader(
            args.batch_size), epoch, args, 'user')
        training(agree, dataset.get_group_dataloader(
            128), epoch, args, 'group')
        print("user and group training time is: [%.1f s]" % (time() - t1))
        t2 = time()

        # evaluation
        u_hrs, u_ndcgs = [0, 0, 0, 0], [0, 0, 0, 0]
        hrs, ndcgs = evaluation(
            agree, helper, dataset.group_testRatings, dataset.group_testNegatives, 'group')
        for i, _ in enumerate(tops):
            print('top%d:Group Iteration %d [%.1f s]: HR = %.4f, '
                  'NDCG = %.4f, [%.1f s]' % (tops[i], epoch, time() - t1, hrs[i], ndcgs[i], time() - t2))

        history.append([u_hrs, u_ndcgs, hrs, ndcgs])
