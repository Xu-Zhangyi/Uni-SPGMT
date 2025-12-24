import numpy as np
import os.path as osp
from tqdm import tqdm
from setting import SetParameter
import random
config = SetParameter()
random.seed(1933)
np.random.seed(1933)
if config.dataset == 'beijing':
    point_num = 112557  # tdrive: 74671; beijing: 112557; porto: 128466
    matrix_num = 113000  # tdrive: 75000; beijing: 113000; porto: 129000
    LCRS_num = 7250
elif config.dataset == 'porto':
    point_num = 128466
    matrix_num = 129000
    LCRS_num = 3286


def re_matrix(traj_list, coor_list, input_dis_matrix, l):
    idx = []
    for i in tqdm(range(len(input_dis_matrix))):
        if np.sum(input_dis_matrix[i] >= 0) > l:
            idx.append(i)
    idx = np.array(idx)
    out_traj_list = traj_list[idx]
    out_coor_list = coor_list[idx]
    out_dis_matrix = input_dis_matrix[idx.reshape(-1, 1), idx.reshape(1, -1)]

    return out_traj_list, out_coor_list, out_dis_matrix


def get_label(input_dis_matrix, count):
    label = []
    for i in tqdm(range(len(input_dis_matrix))):  # (5000,5000)
        input_r = np.array(input_dis_matrix[i])
        # input_r = input_r[np.where(input_r != -1)[0]]
        idx = np.argsort(input_r)
        # label.append(idx[1])
        val = input_r[idx]
        idx = idx[val != -1]
        if len(idx) < 51:
            print(idx)
        label.append(idx[1:count+1])
    return np.array(label)


def get_train_label(re_train_node, re_train_coor, input_dis_matrix, count):

    node_list = []
    coor_list = []

    traj_idx = []
    traj_dis = []

    for i in tqdm(range(len(input_dis_matrix))):
        input_r = np.array(input_dis_matrix[i])
        # input_r = input_r[np.where(input_r != -1)[0]]
        idx = np.argsort(input_r)
        # label.append(idx[1])
        val = input_r[idx]
        re_idx = idx[val != -1]
        re_val = val[val != -1]

        traj_idx.append(re_idx)
        traj_dis.append(re_val)

        node_list.append(re_train_node[i])
        coor_list.append(re_train_coor[i])

    node_list = np.array(node_list, dtype=object)
    coor_list = np.array(coor_list, dtype=object)
    traj_idx = np.array(traj_idx, dtype=object)
    traj_dis = np.array(traj_dis, dtype=object)

    return node_list, coor_list, traj_idx, traj_dis


def select_subgraph_by_degree(dis_matrix, node_list, coor_list, k, l):
    missing_dis = -1
    # degree is count of entries that are not the missing marker, excluding self-loop
    deg = np.sum(dis_matrix != missing_dis, axis=1) - 1
    valid_idx = np.where(deg > l)[0]
    if len(valid_idx) == 0:
        return node_list, coor_list, dis_matrix
    # sort by degree descending
    sorted_idx = valid_idx[np.argsort(deg[valid_idx])[::-1]]
    pick = sorted_idx[:k] if len(sorted_idx) >= k else sorted_idx
    # induce subgraph
    pick_col = pick.reshape(1, -1)
    pick_row = pick.reshape(-1, 1)
    sub_dis = dis_matrix[pick_row, pick_col]
    sub_node = node_list[pick]
    sub_coor = coor_list[pick]
    # ensure within subgraph each node has degree > l
    sub_deg = np.sum(sub_dis != missing_dis, axis=1) - 1
    # iteratively remove nodes not meeting degree until all satisfy or cannot
    while True:
        bad = np.where(sub_deg <= l)[0]
        if len(bad) == 0:
            break
        # remove bad nodes
        keep = np.array([i for i in range(len(sub_node)) if i not in set(bad)])
        if len(keep) == 0:
            break
        sub_dis = sub_dis[keep.reshape(-1, 1), keep.reshape(1, -1)]
        sub_node = sub_node[keep]
        sub_coor = sub_coor[keep]
        sub_deg = np.sum(sub_dis != missing_dis, axis=1) - 1
    return sub_dis, sub_node, sub_coor


all_node_list_int = np.load(config.shuffle_node_file, allow_pickle=True)
all_coor_list_int = np.load(config.shuffle_coor_file, allow_pickle=True)

print(config.dataset)
print(config.distance_type)
train_node = all_node_list_int[0:config.train_set_size]
vali_node = all_node_list_int[config.train_set_size:config.vali_set_size]
test_node = all_node_list_int[config.vali_set_size:config.test_set_size]

train_coor = all_coor_list_int[0:config.train_set_size]
vali_coor = all_coor_list_int[config.train_set_size:config.vali_set_size]
test_coor = all_coor_list_int[config.vali_set_size:config.test_set_size]


train_dis_matrix = np.load(osp.join('/data', config.dataset,
                           config.distance_type, 'train_spatial_distance_50000.npy'))

# vali_dis_matrix = np.load(osp.join('/data', config.dataset,
#                           config.distance_type, 'vali_spatial_distance_50000.npy'))
# test_dis_matrix = np.load(osp.join('/data', config.dataset,
#                            config.distance_type, 'test_spatial_distance_50000.npy'))

np.fill_diagonal(train_dis_matrix, 0)
# np.fill_diagonal(vali_dis_matrix, 0)
# np.fill_diagonal(test_dis_matrix, 0)

if config.distance_type == 'LCRS':
    train_dis_matrix[train_dis_matrix == LCRS_num] = -1


re_train_node, re_train_coor, re_train_dis_matrix = re_matrix(
    train_node, train_coor, train_dis_matrix, config.pos_num*2.5)

# re_vali_node, re_vali_coor, re_vali_dis_matrix = re_matrix(vali_node, vali_coor, vali_dis_matrix, 50)
# re_test_node, re_test_coor, re_test_dis_matrix = re_matrix(test_node, test_coor, test_dis_matrix, 50)

norm_dis = np.max(re_train_dis_matrix)
re_train_dis_matrix = re_train_dis_matrix / norm_dis
re_train_dis_matrix[re_train_dis_matrix < 0] = -1


if config.use_less_traj:
    if config.distance_type == 'LCRS':
        re_train_dis_matrix, re_train_node, re_train_coor = select_subgraph_by_degree(
            re_train_dis_matrix, re_train_node, re_train_coor, 500, config.pos_num*1.5)
    else:
        re_train_node = re_train_node[:500]
        re_train_coor = re_train_coor[:500]
        re_train_dis_matrix = re_train_dis_matrix[:500, :500]
        re_train_node, re_train_coor, re_train_dis_matrix = re_matrix(
            re_train_node, re_train_coor, re_train_dis_matrix, config.pos_num*2)

re_node_list, re_coor_list, traj_idx, traj_dis = get_train_label(re_train_node,
                                                                 re_train_coor,
                                                                 re_train_dis_matrix,
                                                                 config.pos_num)
# vali_y = get_label(re_vali_dis_matrix, 50)
# test_y = get_label(re_test_dis_matrix, 50)
print('train set size:', len(re_node_list))
np.savez(config.spatial_train_set,
         train_node=re_node_list,
         train_coor=re_coor_list,
         traj_idx=traj_idx,
         traj_dis=traj_dis,
         norm_dis=norm_dis)
# np.savez(config.spatial_vali_set,
# vali_node=re_vali_node,
# vali_coor=re_vali_coor,
# vali_y=vali_y)
# np.savez(config.spatial_test_set,
#  vali_node=re_test_node,
#  vali_coor=re_test_coor,
#  vali_y=test_y)
