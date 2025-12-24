import logging
import os.path as osp
import torch
import os
import numpy as np
import random
from typing import Optional, Dict, Any


class SetParameter:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SetParameter, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.dataset: str = 'porto'  # beijing,porto,tdrive
        self.distance_type: str = 'TP'  # 'TP','DITA','discret_frechet','LCRS','NetERP'
        self.cuda: str = '0'  # '0','1', '2', '3'
        self.use_less_traj: bool = True  # 是否使用少量轨迹
        self.save_folder: Optional[str] = None
        self.use_pretrain: bool = True


        if self.dataset == 'tdrive':
            self.dataset_size = 30000
            self.train_set_size = 10000
            self.vali_set_size = 14000
            self.test_set_size = 30000
        elif self.dataset in ('beijing', 'porto'):
            self.dataset_size = 50000
            self.train_set_size = 15000
            self.vali_set_size = 20000
            self.test_set_size = 50000

        self.data_dir = '/data'
        # 节点和边信息
        self.node_file = osp.join('dataset', self.dataset, 'node.csv')
        self.edge_file = osp.join('dataset', self.dataset, 'edge_weight.csv')
        # 距离矩阵
        self.distance_matrix = osp.join(self.data_dir, self.dataset, 'Point_dis_matrix.npy')

        # knn邻居
        if self.dataset == 'beijing':
            self.knn_neighbor_num = 200  
            self.knn_distance_coe = 5000  
        elif self.dataset == 'porto':
            self.knn_neighbor_num = 200
            self.knn_distance_coe = 2000  
        elif self.dataset == 'tdrive':
            self.knn_neighbor_num = 200
            self.knn_distance_coe = 50 
        self.knn_neighbor_file = osp.join('ground_truth', self.dataset, 'knn_neighbor.npy')  
        self.knn_distance_file = osp.join('ground_truth', self.dataset, 'knn_distance.npy')
        # 原始轨迹
        self.traj_file = osp.join('dataset', self.dataset, 'matching_result.pt')
        self.tdrive_traj_file = osp.join('dataset', self.dataset, 'matching_result.csv')
        self.tdrive_time_file = osp.join('dataset', self.dataset, 'time_drop_list.csv')

        self.data_file = osp.join('data', self.dataset, 'st_traj')
        # 随机筛选之后的空间轨迹
        self.shuffle_node_file = osp.join(self.data_file, 'shuffle_node_list.npy')
        self.shuffle_coor_file = osp.join(self.data_file, 'shuffle_coor_list.npy')
        self.shuffle_kseg_file = osp.join(self.data_file, 'shuffle_kseg_list.npy')
        self.shuffle_index_file = osp.join(self.data_file, 'shuffle_index_list.npy')

        # self.shuffle_time_file = osp.join(self.data_file, 'shuffle_time_list.npy')
        self.shuffle_d2vec_file = osp.join(self.data_file, 'shuffle_d2vec_list.npy')

        # label标签
        if self.use_less_traj:
            self.spatial_train_set = osp.join('ground_truth', self.dataset,
                                              self.distance_type, 'spatial_train_set_500.npz')
        else:
            self.spatial_train_set = osp.join('ground_truth', self.dataset,
                                              self.distance_type, 'spatial_train_set.npz')
        self.spatial_vali_set = osp.join('ground_truth', self.dataset,
                                         self.distance_type, 'spatial_vali_set.npz')
        self.spatial_test_set = osp.join('ground_truth', self.dataset,
                                         self.distance_type, 'spatial_test_set.npz')

        # self.spatial_vali_set_pre = osp.join('ground_truth', self.dataset, 'spatial_vali_set_pre.npz')
        self.spatial_long_test_set = osp.join('ground_truth', self.dataset,
                                              self.distance_type, 'spatial_long_test_set.npz')
        self.spatial_num_test_set = osp.join('ground_truth', self.dataset,
                                             self.distance_type, 'spatial_num_test_set.npz')

        self.pointnum: Dict[str, int] = {
            'beijing': 113000,
            'porto': 129000,
            'tdrive': 75000
        }
        self.truenum: Dict[str, int] = {
            'beijing': 112557,
            'porto': 128466,
            'tdrive': 74671
        }
        self.kseg: int = 5
        self.pos_num: int = 10

        self.alpha1: float = 0.75
        self.alpha2: float = 0.25
        self.num_knn: int = 5
        self.feature_size: int = 64
        self.embedding_size: int = 64
        self.date2vec_size: int = 64
        self.hidden_size: int = 64
        self.num_layers: int = 1
        self.dropout_rate: float = 0.3

        if self.use_less_traj and self.distance_type in ['LCRS', 'DITA']:
            self.train_learning_rate: float = 0.0001
        else:
            self.train_learning_rate: float = 0.001

        self.pretrain_learning_rate: float = 0.001

        self.concat: bool = False
        self.train_epochs: int = 60
        self.pretrain_epochs: int = 60
        self.early_stop: int = 50
        self.pretrain_node_batch: int = 1600
        if self.dataset == 'beijing':
            self.TrajMinLen: int = 21
            self.pretrain_traj_batch: int = 40
        elif self.dataset == 'porto':
            self.TrajMinLen: int = 11
            self.pretrain_traj_batch: int = 30
        elif self.dataset == 'tdrive':
            self.TrajMinLen: int = 10
            self.pretrain_traj_batch: int = 40

        self.train_batch: int = 20
        self.test_batch: int = 128

        if self.dataset == 'beijing':
            self.select_pretrain_epoch: int = 35
        elif self.dataset == 'porto':
            self.select_pretrain_epoch: int = 25
        elif self.dataset == 'tdrive':
            self.select_pretrain_epoch: int = 8

        self.pretrain: Dict[str, Any] = {
            'useGraphNodeDist': True,
            'useGraphCoorPred': True,

            'useTrajAutoRegNode': True,
            'useTrajAutoRegCoor': True,

            'useTrajEmbDist': True,

            'TrajNum': 4000,
            'TrajMinLen': self.TrajMinLen,
            'TrajMaxLen': 400,
            'TrajSamNum': 20
        }
        if self.use_pretrain:
            self.use_pretrain_model: Dict[str, Any] = {
                'usePreGraphEncoder': True,
                'usePreTrajEncoder': True,
            }
        else:
            self.use_pretrain_model: Dict[str, Any] = {
                'usePreGraphEncoder': False,
                'usePreTrajEncoder': False,
            }

        self.gtraj: Dict[str, Any] = {
            'usePI': True,
            'useSI': True,
            'useTransPI': True,
            'useTransGI': True,
        }
        self.node2vec: Dict[str, Any] = {
            'walk_length': 20,
            'context_size': 10,
            'walks_per_node': 10,
            'num_neg_samples': 1,
            'p': 1,
            'q': 1
        }
        self.device = torch.device('cuda:' + self.cuda) if torch.cuda.is_available() else torch.device('cpu')

        if str(self.distance_type) == "TP":
            self.coe = 8*4
        elif str(self.distance_type) == "DITA":
            if str(self.dataset) == "beijing":
                self.coe = 32*2
            elif str(self.dataset) == "porto":
                self.coe = 16
        elif str(self.distance_type) == "LCRS":
            if str(self.dataset) == "beijing":
                self.coe = 4*16*2
            elif str(self.dataset) == "porto":
                self.coe = 2*16
        elif str(self.distance_type) == "discret_frechet":
            self.coe = 8*2

        self.pretrain_coe = 1

        allowed_dis = {'TP', 'DITA', 'discret_frechet', 'LCRS'}
        if self.distance_type not in allowed_dis:
            raise ValueError(f"Invalid value for distance_type. Allowed values are: {allowed_dis}")
        allowed_data = {'beijing', 'porto', 'tdrive'}
        if self.dataset not in allowed_data:
            raise ValueError(f"Invalid value for dataset. Allowed values are: {allowed_data}")

    def log_parameters(self):

        # 基本参数
        logging.info(f"Dataset: {self.dataset}")
        logging.info(f"Distance Type: {self.distance_type}")
        logging.info(f"CUDA Device: {self.cuda}")
        logging.info(f"Save Folder: {self.save_folder}")

        logging.info(f"Positive Number: {self.pos_num}")

        # 模型参数
        logging.info(f"Alpha1: {self.alpha1}")
        logging.info(f"Alpha2: {self.alpha2}")
        logging.info(f"Number of KNN: {self.num_knn}")
        logging.info(f"Feature Size: {self.feature_size}")
        logging.info(f"Embedding Size: {self.embedding_size}")
        logging.info(f"Date2Vec Size: {self.date2vec_size}")
        logging.info(f"Number of Layers: {self.num_layers}")
        logging.info(f"Dropout Rate: {self.dropout_rate}")
        logging.info(f"Train Learning Rate: {self.train_learning_rate}")
        logging.info(f"Pretrain Learning Rate: {self.pretrain_learning_rate}")
        logging.info(f"Train Epochs: {self.train_epochs}")
        logging.info(f"Pretrain Epochs: {self.pretrain_epochs}")
        logging.info(f"Early Stop: {self.early_stop}")

        logging.info(f"Pretrain Node Batch Size: {self.pretrain_node_batch}")
        logging.info(f"Pretrain Traj Batch Size: {self.pretrain_traj_batch}")
        logging.info(f"Train Batch: {self.train_batch}")
        logging.info(f"Test Batch: {self.test_batch}")
        logging.info(f"knn neighbor num: {self.knn_neighbor_num}")
        logging.info(f"knn distance coe: {self.knn_distance_coe}")
        logging.info(f"Coe: {self.coe}")
        logging.info(f"Pretrain Coe: {self.pretrain_coe}")

        # 点数信息
        logging.info(f"Point Numbers: {self.pointnum}")
        logging.info(f"True Numbers: {self.truenum}")

        # 预训练参数
        logging.info("Pretrain Parameters:")
        for key, value in self.pretrain.items():
            logging.info(f"  {key}: {value}")

        logging.info("Use Pretrain Model:")
        for key, value in self.use_pretrain_model.items():
            logging.info(f"  {key}: {value}")

        # GTrajectory参数
        logging.info("GTrajectory Parameters:")
        for key, value in self.gtraj.items():
            logging.info(f"  {key}: {value}")

        # Node2Vec参数
        logging.info("Node2Vec Parameters:")
        for key, value in self.node2vec.items():
            logging.info(f"  {key}: {value}")

        # 文件路径
        logging.info("File Paths:")
        logging.info(f"  Node File: {self.node_file}")
        logging.info(f"  Edge File: {self.edge_file}")
        logging.info(f"  Distance Matrix: {self.distance_matrix}")
        if self.dataset in ('beijing', 'porto'):
            logging.info(f"  Trajectory File: {self.traj_file}")
        if self.dataset == 'tdrive':
            logging.info(f"  TDrive Trajectory File: {self.tdrive_traj_file}")
            logging.info(f"  TDrive Time File: {self.tdrive_time_file}")

        logging.info(f"  Data Directory: {self.data_dir}")
        logging.info(f"  Shuffle Node File: {self.shuffle_node_file}")
        logging.info(f"  Shuffle Coordinate File: {self.shuffle_coor_file}")
        logging.info(f"  Spatial Train Set: {self.spatial_train_set}")
        logging.info(f"  Spatial Validation Set: {self.spatial_vali_set}")
        logging.info(f"  Spatial Test Set: {self.spatial_test_set}")

        logging.info("=" * 60)


def setup_logger(fname: Optional[str] = None) -> None:
    if fname:
        log_dir = osp.dirname(fname)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
    logging.root.handlers.clear()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=fname,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def zipdir(path: str, ziph: Any, include_format: list[str]) -> None:
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)
