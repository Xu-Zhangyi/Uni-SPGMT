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
elif config.dataset == 'tdrive':
    point_num = 74671
    matrix_num = 75000
    LCRS_dis_num = 300
    LCRS_time_num = 300

if str(config.dataset) == "tdrive":
    if str(config.distance_type) == "TP":
        coe = 8
    elif str(config.distance_type) == "DITA":
        coe = 32*4
    elif str(config.distance_type) == "LCRS":
        coe = 4
    elif str(config.distance_type) == "discret_frechet":
        coe = 8*2


def re_matrix(traj_list, coor_list, time_list, input_dis_matrix, input_time_matrix, l):
    idx = []
    for i in tqdm(range(len(input_dis_matrix))):
        if np.sum(input_dis_matrix[i] >= 0) > l:
            idx.append(i)
    idx = np.array(idx)
    out_traj_list = traj_list[idx]
    out_coor_list = coor_list[idx]
    out_time_list = time_list[idx]
    out_dis_matrix = input_dis_matrix[idx.reshape(-1, 1), idx.reshape(1, -1)]
    out_time_matrix = input_time_matrix[idx.reshape(-1, 1), idx.reshape(1, -1)]

    return out_traj_list, out_coor_list, out_time_list, out_dis_matrix, out_time_matrix


def get_label(input_dis_matrix, input_time_matrix, count):
    label = []
    for i in tqdm(range(len(input_dis_matrix))):  # (5000,5000)
        input_r = np.array(input_dis_matrix[i])
        input_time = np.array(input_time_matrix[i])

        out_val = input_r+input_time
        out_val[input_r == -1] = -1

        idx = np.argsort(out_val)
        # label.append(idx[1])
        val = out_val[idx]
        idx = idx[val != -1]  # 升序,第0位是自己
        if len(idx) < 51:
            print(idx)
        label.append(idx[1:count+1])
    return np.array(label)


def get_train_label(input_dis_matrix, input_time_matrix, count):

    traj_idx = []
    traj_dis = []
    for i in tqdm(range(len(input_dis_matrix))):  # (5000,5000)
        input_r = np.array(input_dis_matrix[i])
        input_time = np.array(input_time_matrix[i])

        idx = np.argsort(input_r)
        val = input_r[idx]
        re_idx = idx[val != -1]
        re_val = val[val != -1]  # input_r[re_idx]
        val_time = input_time[re_idx]
        out_val = (re_val+val_time)/2

        traj_idx.append(re_idx)
        traj_dis.append(out_val)

    traj_idx = np.array(traj_idx, dtype=object)
    traj_dis = np.array(traj_dis, dtype=object)

    return traj_idx, traj_dis


all_node_list_int = np.load(osp.join('data', config.dataset, 'st_traj', 'shuffle_node_list.npy'), allow_pickle=True)
all_coor_list_int = np.load(osp.join('data', config.dataset, 'st_traj', 'shuffle_coor_list.npy'), allow_pickle=True)
all_d2vec_list_int = np.load(osp.join('data', config.dataset, 'st_traj', 'shuffle_d2vec_list.npy'), allow_pickle=True)
print(config.dataset)
print(config.distance_type)


train_list = all_node_list_int[0:config.train_set_size]  # [0:10000]
vali_list = all_node_list_int[config.train_set_size:config.vali_set_size]  # [10000:14000]
test_list = all_node_list_int[config.vali_set_size:config.test_set_size]  # [14000:30000]

train_coor = all_coor_list_int[0:config.train_set_size]  # [0:10000]
vali_coor = all_coor_list_int[config.train_set_size:config.vali_set_size]  # [10000:14000]
test_coor = all_coor_list_int[config.vali_set_size:config.test_set_size]  # [14000:30000]

train_d2vec = all_d2vec_list_int[0:config.train_set_size]  # [0:10000]
# vali_d2vec = all_d2vec_list_int[config.train_set_size:config.vali_set_size]  # [10000:14000]
test_d2vec = all_d2vec_list_int[config.vali_set_size:config.test_set_size]  # [14000:30000]

train_dis_matrix = np.load(osp.join('/data/xuzhangyi/SPGMT', config.dataset, config.dataset,
                           config.distance_type, 'train_spatial_distance_30000.npy'))
# vali_dis_matrix = np.load(osp.join('/data/xuzhangyi/SPGMT', config.dataset, config.dataset,
#                           config.distance_type, 'vali_spatial_distance_30000.npy'))
test_dis_matrix = np.load(osp.join('/data/xuzhangyi/SPGMT', config.dataset, config.dataset,
                          config.distance_type, 'test_spatial_distance_30000.npy'))

train_time_matrix = np.load(osp.join('/data/xuzhangyi/SPGMT', config.dataset, config.dataset,
                                     config.distance_type, 'train_temporal_distance.npy'))
# vali_time_matrix = np.load(osp.join('/data/xuzhangyi/SPGMT', config.dataset, config.dataset,
#                                     config.distance_type, 'vali_temporal_distance.npy'))
test_time_matrix = np.load(osp.join('/data/xuzhangyi/SPGMT', config.dataset, config.dataset,
                                    config.distance_type, 'test_temporal_distance.npy'))

# %==========================================================================================================
np.fill_diagonal(train_dis_matrix, 0)
# np.fill_diagonal(vali_dis_matrix, 0)
np.fill_diagonal(test_dis_matrix, 0)
np.fill_diagonal(train_time_matrix, 0)
# np.fill_diagonal(vali_time_matrix, 0)
np.fill_diagonal(test_time_matrix, 0)

if config.distance_type == 'LCRS':
    train_dis_matrix[train_dis_matrix == LCRS_dis_num] = -1
# # # %==========================================================================================================

re_train_node, re_train_coor, re_train_d2vec, re_train_dis_matrix, re_train_time_matrix = re_matrix(
    train_list, train_coor, train_d2vec, train_dis_matrix, train_time_matrix, config.pos_num*2.5)

# re_vali_list, re_vali_coor, re_vali_d2vec, re_vali_dis_matrix, re_vali_time_matrix = re_matrix(
#     vali_list, vali_coor, vali_d2vec, vali_dis_matrix, vali_time_matrix, 50)
re_test_list, re_test_coor, re_test_d2vec, re_test_dis_matrix, re_test_time_matrix = re_matrix(
    test_list, test_coor, test_d2vec, test_dis_matrix, test_time_matrix, 50)
# %=================================================================================
norm_num = np.max(re_train_dis_matrix)
re_train_dis_matrix = re_train_dis_matrix / norm_num * coe
re_train_dis_matrix[re_train_dis_matrix < 0] = -1

# norm_num = np.max(re_vali_dis_matrix)
# re_vali_dis_matrix = re_vali_dis_matrix / norm_num * coe
# re_vali_dis_matrix[re_vali_dis_matrix < 0] = -1

norm_num = np.max(re_test_dis_matrix)
re_test_dis_matrix = re_test_dis_matrix / norm_num * coe
re_test_dis_matrix[re_test_dis_matrix < 0] = -1

norm_num_time = np.max(re_train_time_matrix)
re_train_time_matrix = re_train_time_matrix / norm_num_time * coe
re_train_time_matrix[re_train_time_matrix < 0] = 0

# norm_num_time = np.max(re_vali_time_matrix)
# re_vali_time_matrix = re_vali_time_matrix / norm_num_time * coe
# re_vali_time_matrix[re_vali_time_matrix < 0] = -1

norm_num_time = np.max(re_test_time_matrix)
re_test_time_matrix = re_test_time_matrix / norm_num_time * coe
re_test_time_matrix[re_test_time_matrix < 0] = -1
# %==========================================================================================================


def select_subgraph_by_degree(dis_matrix, time_matrix, node_list, coor_list, d2vec_list, k, l):
    # treat LCRS_dis_num as missing edge marker when available; fallback to -1
    # missing_dis = LCRS_dis_num if 'LCRS_dis_num' in globals() else -1
    missing_dis = -1
    # degree is count of entries that are not the missing marker, excluding self-loop
    deg = np.sum(dis_matrix != missing_dis, axis=1) - 1
    valid_idx = np.where(deg > l)[0]
    if len(valid_idx) == 0:
        return node_list, coor_list, d2vec_list, dis_matrix, time_matrix
    # sort by degree descending
    sorted_idx = valid_idx[np.argsort(deg[valid_idx])[::-1]]
    pick = sorted_idx[:k] if len(sorted_idx) >= k else sorted_idx
    # induce subgraph
    pick_col = pick.reshape(1, -1)
    pick_row = pick.reshape(-1, 1)
    sub_dis = dis_matrix[pick_row, pick_col]
    sub_time = time_matrix[pick_row, pick_col]
    sub_node = node_list[pick]
    sub_coor = coor_list[pick]
    sub_d2vec = d2vec_list[pick]
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
        sub_time = sub_time[keep.reshape(-1, 1), keep.reshape(1, -1)]
        sub_node = sub_node[keep]
        sub_coor = sub_coor[keep]
        sub_d2vec = sub_d2vec[keep]
        sub_deg = np.sum(sub_dis != missing_dis, axis=1) - 1
    return sub_node, sub_coor, sub_d2vec, sub_dis, sub_time


# if config.distance_type == 'LCRS':
#     # 选择1000节点子图，保证每个节点的边数超过 config.pos_num*2
#     re_train_node, re_train_coor, re_train_d2vec, re_train_dis_matrix, re_train_time_matrix = select_subgraph_by_degree(
#         re_train_dis_matrix, re_train_time_matrix, re_train_node, re_train_coor, re_train_d2vec, 500, config.pos_num*1.5)
# else:
#     re_train_node = re_train_node[0:1000]
#     re_train_coor = re_train_coor[0:1000]
#     re_train_d2vec = re_train_d2vec[0:1000]
#     re_train_dis_matrix = re_train_dis_matrix[0:1000, 0:1000]
#     re_train_time_matrix = re_train_time_matrix[0:1000, 0:1000]


# traj_idx, traj_dis = get_train_label(
#     re_train_dis_matrix, re_train_time_matrix, config.pos_num)
# print('train set size:', len(re_train_node))
# print('distance matrix size:', len(traj_idx))
# vali_y = get_label(re_vali_dis_matrix, re_vali_time_matrix, 50)
test_y = get_label(re_test_dis_matrix, re_test_time_matrix, 50)

# np.savez(config.spatial_train_set,
#          train_node=re_train_node,
#          train_coor=re_train_coor,
#          train_d2vec=re_train_d2vec,
#          traj_idx=traj_idx,
#          traj_dis=traj_dis,
#          coe=coe)
# np.savez(config.spatial_vali_set,
#          vali_node=re_vali_list,
#          vali_coor=re_vali_coor,
#          vali_d2vec=re_vali_d2vec,
#          vali_y=vali_y)
np.savez(config.spatial_test_set,
         vali_node=re_test_list,
         vali_coor=re_test_coor,
         vali_d2vec=re_test_d2vec,
         vali_y=test_y)
