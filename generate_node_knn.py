import numpy as np
from tqdm import tqdm
from setting import SetParameter
from collections import Counter
config = SetParameter()


knn_num = 1000
point_dis = np.load(config.distance_matrix)
for i in range(len(point_dis)):
    point_dis[i][i] = 0.0
# 找寻节点最近target节点ß
knn_neighbor = []
knn_distance = []
for i in tqdm(range(config.truenum[config.dataset])):
    sorted_list = np.argsort(point_dis[i])  
    sorted_zero_id = np.argwhere(sorted_list == i)[0][0]  

    # 取出距离为0的节点之后的前1000个节点
    knn_neighbor.append(
        sorted_list[sorted_zero_id+1:sorted_zero_id + knn_num + 1])
    knn_distance.append(
        point_dis[i][sorted_list[sorted_zero_id+1:sorted_zero_id + knn_num + 1]])

np.save(f'ground_truth/{config.dataset}/knn_neighbor',
        np.array(knn_neighbor, dtype=object))
np.save(f'ground_truth/{config.dataset}/knn_distance',
        np.array(knn_distance, dtype=object))


all_node_knn_neighbor = np.load(
    f'ground_truth/{config.dataset}/knn_neighbor.npy', allow_pickle=True)
all_node_knn_distance = np.load(
    f'ground_truth/{config.dataset}/knn_distance.npy', allow_pickle=True)
k5_neighbor = []
k5_distance = []
k10_neighbor = []
k10_distance = []
k15_neighbor = []
k15_distance = []
for i in tqdm(range(config.truenum[config.dataset])):
    i_knn_neighbor = all_node_knn_neighbor[i]
    i_knn_distance = all_node_knn_distance[i]
    for idx, j in enumerate(i_knn_neighbor[0:5]):
        k5_neighbor.append(np.array([i, j]))
        k5_distance.append(i_knn_distance[idx])
    for idx, j in enumerate(i_knn_neighbor[5:10]):
        k10_neighbor.append(np.array([i, j]))
        k10_distance.append(i_knn_distance[idx + 5])
    for idx, j in enumerate(i_knn_neighbor[10:15]):
        k15_neighbor.append(np.array([i, j]))
        k15_distance.append(i_knn_distance[idx + 10])

np.save(f'./dataset/{config.dataset}/5/k5_neighbor', np.array(k5_neighbor))
np.save(f'./dataset/{config.dataset}/5/k5_distance', np.array(k5_distance))
np.save(f'./dataset/{config.dataset}/5/k10_neighbor', np.array(k10_neighbor))
np.save(f'./dataset/{config.dataset}/5/k10_distance', np.array(k10_distance))
np.save(f'./dataset/{config.dataset}/5/k15_neighbor', np.array(k15_neighbor))
np.save(f'./dataset/{config.dataset}/5/k15_distance', np.array(k15_distance))
print('successful!')
