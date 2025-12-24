# Uni-SPGMT

Source codes for Uni-SPGMT

## Running Procedures:

1. Extract the dataset folder, you can get the 'matching_result.pt', 'node.csv', and 'edge_weight.csv' of Beijing or Porto dataset.
   
   Example: `dataset/beijing/matching_result.pt`

2. Run 'spatial_preprocess.py' to obtain the initial structural embeddings 'node_features.npy' for trajectories as well as 'shuffle_node_list.npy', 'shuffle_coor_list.npy', 'shuffle_kseg_list.npy'.
   
   Example: `data/beijing/node_features.npy`, `data/beijing/st_traj/shuffle_node_list.npy`

3. Run 'spatial_similarity_computation.py' to compute the pairwise point distances and ground truth similarities for trajectories. This will take some time.
    
    Example: `ground_truth/beijing/Point_dis_matrix.npy`, `ground_truth/beijing/TP/train_spatial_distance_50000.npy`

4. Run 'generate_trajectory.py' to obtain the training set, validation set, and test set. We only use a small number of trajectories for fine-tuning.

    Example: `ground_truth/beijing/TP/spatial_train_set_500.npz`

5. Run 'generate_node_knn.py' to get the kNN neighbors for each node in the road network.

    Example: `dataset/beijing/5/k5_neighbor.npy`

6. Run 'spatial_data_utils.py' to obtain the position embeddings.

    Example: `dataset/beijing/distance_to_anchor_node_10000.pt`

7. Run 'main.py'.

   - **Pre-training phase**:
     ```python
     GNodeSim = GraphPretrain_Trainer()
     GNodeSim.Spa_train()
     ```
     Obtains the pre-trained model `GraphNodeSimEncoder`.

   - **Fine-tuning phase**: 
     ```python
     GTrajSim = GTrajSim_Trainer()
     GTrajSim.Spa_train()
     ```
     Loads `GraphNodeSimEncoder` and performs fine-tuning to obtain the final model.

   - **Testing phase**:
     ```python
     GTrajSim = GTrajSim_Trainer()
     GTrajSim.Spa_eval()
     ```
