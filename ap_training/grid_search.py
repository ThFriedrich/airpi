"""Hyperparameter Grid Search"""
import os
import argparse
import numpy as np
from tensorflow import keras
from airpi.ap_architectures.utils import PRM, checkHyperparametersExist, handleArgs
from airpi.ap_training.data_fcns import getDatasets
from airpi.ap_training.callbacks import getCallbacks
from airpi.ap_architectures.unet import UNET
import airpi.ap_training.losses as ls


def fcn_train(prms_o, prms_net, first):
    """Train for a given Setup"""

    training_ds, validation_ds = getDatasets(prms_o, True)

    #####################
    #     Callbacks     #
    #####################
    # TensorBoard write interval
    tb_freq = 2
    # tb_freq = P.X_SHAPE[0] // prms_o['batch_size'] // 30
    callbacks = getCallbacks(prms_o, prms_net, tb_freq, 1, 1)

    #####################
    #       Model       #
    #####################

    # model = CNN2D(P.X_SHAPE, prms_net).model
    model = UNET(PRM.X_SHAPE, prms_net).model

    loss_fcn = ls.loss(prms_o)
    metric_fcn = ls.metric_fcns(prms_o, model)

    model.compile(
        optimizer=keras.optimizers.SGD(prms_o['learning_rate'], momentum=float(prms_o['m_optimizer']), nesterov=True),
        # optimizer=keras.optimizers.Adam(prms_o['learning_rate'], beta_1=float(prms_o['m_optimizer'])),
        loss=loss_fcn,
        metrics=[metric_fcn])

    if os.path.exists(os.path.join(os.getcwd(), prms_o['cp_path'], 'initial_weights.h5')) is False:
        model.save_weights(os.path.join(
            os.getcwd(), prms_o['cp_path'], 'initial_weights.h5'))
    else:
        model.load_weights(os.path.join(
            os.getcwd(), prms_o['cp_path'], 'initial_weights.h5'))

    model.fit(training_ds,
              validation_data=validation_ds,
              validation_steps=prms_o['validation_steps'],
              validation_freq=1,
              epochs=prms_o['epochs'],
              steps_per_epoch=prms_o['steps_per_epoch'],
              initial_epoch=prms_o['ep'],
              shuffle=False,
              callbacks=callbacks)


def batch_lr(prms_o, prms_net, batch_sizes, momentums):
    """Loops over given Parameters"""
    cou = 0
    for ba in batch_sizes:
        prms_o.update({'batch_size': ba})
        for ma in momentums:
            prms_o.update({'ep': 0})
            prms_o.update({'m_optimizer': ma})
            fcn_train(prms_o, prms_net, cou == 0)
            cou += 1

###################################################################################
if __name__ == '__main__':
    os.system('clear')

    parser = argparse.ArgumentParser()

    # parse arguments
    parser.add_argument('--model', type=str, default='dbg_', help='Model Name')
    parser.add_argument('--weights', type=float, nargs=5,
                        default=[1.0, 1.0, 1.0, 1.0, 0.0], help='weight factors [L1, sdim]')
    parser.add_argument('--bs', type=int, default=8, help='Batch Size')
    parser.add_argument('--lr', type=int, nargs=3, default=[-5, -3, 20], help='Learning Rate (Parameters for np.logspace)')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu index')
    parser.add_argument('--machine', type=str, choices=['a', 'c', 'd', 'l', 'm'], default='d', help='Running on: a:Aurora, c:Cluster, d:Dell, l:Lenovo')


    args = vars(parser.parse_args())
    model_name, db_location  = handleArgs(args)

    learning_rate_rng = np.logspace(args['lr'][0], args['lr'][1], args['lr'][2])

    # momentum_rng = np.linspace(0.5, 0.99, 10)
    momentum_rng = [0.9]

    batch_size_rng = [args['bs']]

    ds = 'db_h5_3x3_uint16'
    prms = {
        # Data
        'train_path':[  os.path.join(db_location, ds + '_Training.h5'), 
                        os.path.join(db_location, ds + '_Training_2.h5'),
                        os.path.join(db_location, ds + '_Training_3.h5'),
                        os.path.join(db_location, ds + '_Training_4.h5')],
        'val_path':  [  os.path.join(db_location, ds + '_Validation.h5')],
        'test_path': [  os.path.join(db_location, ds + '_Test.h5')],
        'log_path': os.path.join(os.getcwd(), 'Logs', 'Grid_Search', model_name),
        'cp_path': os.path.join(os.getcwd(), 'Ckp', 'Training', model_name),
        # Learning Rate schedule parameters
        'learning_rate': learning_rate_rng[0],
        'learning_rate_0': learning_rate_rng[0],
        'learning_rate_rng': learning_rate_rng,
        'epochs': args['lr'][2],
        'ep': 0,
        'epochs_cycle_1': 1,
        'epochs_cycle': 0,
        'epochs_ramp': 0,
        'warmup': False,
        'cooldown': False,
        'lr_fact': 0.75,
        # Optimizer, Model Parameters
        'loss_prms': args['weights'], # [Pixel_L1, SSIM]
        'batch_size': batch_size_rng[0],
        'num_parallel_calls': 2,
        'norm': 1, #1:L1, 2:L2, 3: Logcosh
        'm_optimizer': 0.9
    }

    # U-NET
    prms_net = {
        'branching': 0,  # 0 = No branching, 1 = fully seperated branches, 2 = Hybrid
        'kernel': [3, 3],
        'normalization': 0,  # 0 = None, 1 = 'instance', 2 = layer else Batch
        'activation': 1,  # 0 = None, 1 = 'LeakyReLU', 2 = ELU, 3 = SWISH else ReLu
        'filters': 32,
        'depth': 4,
        'dcr': True,
        'w_regul': None,
        'a_regul': None,
        'dropout': None,
    }


    PRM.read_parameters(prms)
    prms, prms_net, b_loaded = checkHyperparametersExist(prms, prms_net)
    batch_lr(prms, prms_net, batch_size_rng,
             momentum_rng)
