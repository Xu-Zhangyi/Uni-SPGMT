from setting import SetParameter, set_seed, setup_logger, zipdir
import datetime
import os.path as osp
import os
import zipfile
import pathlib
import logging
from Trainer import GraphPretrain_Trainer, GTrajSim_Trainer
from TrainerST import GraphPretrainST_Trainer, GTrajST_Trainer
# set_seed(11)
config = SetParameter()

if __name__ == '__main__':

    # -----------pretrain set-----------

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(osp.join('log_pretrain', config.dataset)):
        os.makedirs(osp.join('log_pretrain', config.dataset))
    config.save_folder = osp.join('saved_models_pretrain', config.dataset, f'{str(current_time)}')
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    setup_logger(osp.join('log_pretrain', config.dataset, f'lstm.{str(current_time)}_pretrain.log'))
    config.log_parameters()
    zipf = zipfile.ZipFile(os.path.join(config.save_folder, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(str(pathlib.Path().absolute()), zipf, include_format=['.py'])
    zipf.close()
    # -----------pretrain SPGMT-----------
    GNodeSim = GraphPretrain_Trainer()
    GNodeSim.Spa_train()

    # %====================================================================

    # # -----------train set------------------------------------------------------
    # current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # if not os.path.exists(osp.join('log', config.dataset)):
    #     os.makedirs(osp.join('log', config.dataset))
    # config.save_folder = osp.join('saved_models', config.dataset, config.distance_type + '_' + str(current_time))
    # if not os.path.exists(config.save_folder):
    #     os.makedirs(config.save_folder)
    # setup_logger(osp.join('log', config.dataset,f'lstm.{config.distance_type}_{str(current_time)}_train.log'))
    # config.log_parameters()
    # zipf = zipfile.ZipFile(os.path.join(config.save_folder, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    # zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    # zipf.close()
    # # # -----------train SPGMT-----------
    # if config.use_pretrain:
    #     # If using pre-trained model, load it
    #     pretrain_time = ''  # 
    #     select_pretrain_epoch= config.select_pretrain_epoch  
    #     pretrain_net_path = osp.join('saved_models_pretrain', config.dataset,
    #                                 str(pretrain_time), f'GraphNodeSimEncoder_{config.dataset}_epochs{select_pretrain_epoch}.pth')
    #     logging.info(f'Using pre-trained model from: {pretrain_net_path}')
    #     GTrajSim = GTrajSim_Trainer(config.save_folder, pretrain_folder=pretrain_net_path)
    # else:
    #     logging.info('Not using pre-trained model.')
    #     GTrajSim = GTrajSim_Trainer(config.save_folder, pretrain_folder=None)
    # GTrajSim.Spa_train()

    # # -----------test set-----------
    # current_time = ''
    # epoch_num = ''
    # load_model_name = osp.join('saved_models', config.dataset, config.distance_type +
    #                            '_' + current_time, f'epoch_{epoch_num}.pt')
    # setup_logger(osp.join('log', config.dataset, f'lstm.{config.distance_type}_{str(current_time)}_test_{epoch_num}.log'))
    # config.log_parameters()

    # # # -----------testSPGMT-----------
    # GTrajSim = GTrajSim_Trainer()
    # GTrajSim.Spa_eval(load_model=load_model_name, is_long_traj=False)


    #============================================================================

    # -----------pretrain set-----------

    # current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # if not os.path.exists(osp.join('log_pretrain', config.dataset)):
    #     os.makedirs(osp.join('log_pretrain', config.dataset))
    # config.save_folder = osp.join('saved_models_pretrain', config.dataset, f'{str(current_time)}')
    # if not os.path.exists(config.save_folder):
    #     os.makedirs(config.save_folder)

    # setup_logger(osp.join('log_pretrain', config.dataset, f'lstm.{str(current_time)}_pretrain.log'))
    # config.log_parameters()
    # zipf = zipfile.ZipFile(os.path.join(config.save_folder, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    # zipdir(str(pathlib.Path().absolute()), zipf, include_format=['.py'])
    # zipf.close()
    # # -----------pretrain SPGMT-----------
    # GNodeSim = GraphPretrainST_Trainer()
    # logging.info('mainST')
    # GNodeSim.ST_train()

    # %====================================================================

    # # -----------train set------------------------------------------------------
    # current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # if not os.path.exists(osp.join('log', config.dataset)):
    #     os.makedirs(osp.join('log', config.dataset))
    # config.save_folder = osp.join('saved_models', config.dataset, config.distance_type + '_' + str(current_time))
    # if not os.path.exists(config.save_folder):
    #     os.makedirs(config.save_folder)
    # setup_logger(osp.join('log', config.dataset,f'lstm.{config.distance_type}_{str(current_time)}_train.log'))
    # config.log_parameters()
    # zipf = zipfile.ZipFile(os.path.join(config.save_folder, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    # zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    # zipf.close()
    # config.is_transferability = False
    # # -----------train SPGMT-----------
    # if config.use_pretrain:
    #     # If using pre-trained model, load it
    #     pretrain_time = ''  # 预训练的时间
    #     pretrain_net_path = osp.join('saved_models_pretrain', config.dataset,
    #                                 str(pretrain_time), f'GraphNodeSimEncoder_{config.dataset}_epochs8.pth')
    #     logging.info(f'Using pre-trained model from: {pretrain_net_path}')
    #     GTrajSim = GTrajST_Trainer(config.save_folder, pretrain_folder=pretrain_net_path)
    # else:
    #     logging.info('Not using pre-trained model.')
    #     GTrajSim = GTrajST_Trainer(config.save_folder, pretrain_folder=None)
    # GTrajSim.ST_train()

    #%=======================================================


    # -----------test set-----------
    # current_time = ''
    # epoch_num = ''
    # load_model_name = osp.join('saved_models', config.dataset, config.distance_type +
    #                            '_' + current_time, f'epoch_{epoch_num}.pt')
    # setup_logger(osp.join('log', config.dataset, f'lstm.{config.distance_type}_{str(current_time)}_test_{epoch_num}_2.log'))
    # config.log_parameters()

    # # -----------test SPGMT-----------
    # GTrajSim = GTrajST_Trainer()
    # GTrajSim.ST_eval(load_model=load_model_name, is_long_traj=False, is_transferability=config.is_transferability)
