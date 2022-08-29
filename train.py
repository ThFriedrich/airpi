"""Module for Tensorflow Training"""
import os
import argparse
import silence_tensorflow.auto
import shutil

def fcn_train(prms_o, prms_net, NET):
    '''Train the Network'''
    from tensorflow.keras import optimizers
    from tensorflow.data.experimental import AUTOTUNE
    from ap_training.data_fcns import getDatasets
    import ap_training.losses as ls

    training_ds, validation_ds, validation_steps, steps_per_epoch = getDatasets(prms_o)

    model = NET(prms_net, training_ds._flat_shapes[0][1:4]).build()
    model.summary()

    if os.path.isfile(os.path.join(prms_o['cp_path'], 'checkpoint')) is True:
        model.load_weights(prms_o['cp_path'] + '/cp-best.ckpt')
        print('Resuming training from Epoch ' + str(prms['ep']))

    model.compile(
        optimizer=optimizers.Adam(prms_o['learning_rate']),loss=ls.loss, metrics=ls.metrics())
       
    callbacks = airpi_callbacks(prms_o, prms_net).as_list
    shutil.copy('./ap_training/losses.py', prms_o['cp_path'])
    os.rename(prms_o['cp_path']+'/losses.py', prms_o['cp_path']+'/losses_'+str(prms_o['ep'])+'.py')
    model.fit(training_ds,
              validation_data=validation_ds,
              validation_steps=validation_steps,
              validation_freq=1,
              epochs=prms_o['epochs'],
              steps_per_epoch=steps_per_epoch,
              initial_epoch=prms_o['ep'],
              callbacks=callbacks,
              use_multiprocessing=True,
              workers=AUTOTUNE)


###################################################################################


if __name__ == '__main__':
    os.system('clear')
    parser = argparse.ArgumentParser()

    # parse arguments
    parser.add_argument('--model_name', type=str, help='Model Name')
    parser.add_argument('--arch', type=str, help='Architecture: UNET or COMPLEX')
    parser.add_argument('--filters', type=int, help='UNET filters')
    parser.add_argument('--depth', type=int, help='UNET levels')
    parser.add_argument('--stack_n', type=int, help='UNET conv layers per stack')
    parser.add_argument('--db_path', type=str, help='path to datasets')
    parser.add_argument('--loss_prms_r', type=float, nargs=7, help='weight factors [px_p, px_a, px_pw, px_int, obj]') # the first parameter decide whether use data in r space for training, the rests specifies the weight
    parser.add_argument('--loss_prms_k', type=float, nargs=7, help='weight factors [px_p, px_a, px_pw, px_int, int_tot, df_ratio]') # the first parameter decide whether use data in k space for training, the rests specifies the weight
    parser.add_argument('--bs', type=int, help='Batch Size')
    parser.add_argument('--lr', type=float, help='Learning Rate')
    parser.add_argument('--dose', type=float, nargs=3, help='Dose (logspace(a,b)*c)')
    parser.add_argument('--gpu_id', type=int, help='gpu index')
    parser.add_argument('--branching', type=int, help='For UNET only branching 0, 1 or 2')
    parser.add_argument('--normalization', type=int, help='For UNET only, 0 = None, 1 = instance, 2 = layer else Batch')
    parser.add_argument('--activation', type=int, help='For UNET only 0 = None, 1 = LeakyReLU, 2 = ELU, 3 = SWISH , 4 = Sigmoid else ReLu')
    parser.add_argument('--type', type=str, help='For UNET only V=Vanilla, RES=Residual, DCR=Dense-Conv-Residual')
    parser.add_argument('--epochs', type=int, help='Epochs to train')
    parser.add_argument('--ep', type=int, help='Epoch to start from')
    parser.add_argument('--epochs_cycle_1', type=int, help='Epochs to train for cycle 1')
    parser.add_argument('--epochs_cycle', type=int, help='Epochs to train for all other cycles')
    parser.add_argument('--sample_dir', type=str, help='Directory containing sample datasets')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--global_skip', type=int, help='Add skip connection of probe function to NN output')
    parser.add_argument('--global_cat', type=int, help='Concatenate probe function to last conv-layer input')
    args = vars(parser.parse_args())

    if args["gpu_id"] is None:
        args["gpu_id"] = 1
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])


    from tensorflow import config as tf_config
    tf_config.optimizer.set_jit("autoclustering")

    from ap_utils.file_ops import manage_args
    from ap_utils.globals import debugger_is_active
    prms, prms_net = manage_args(args) 
    
    from ap_training.callbacks import airpi_callbacks
    if prms_net['arch'] == 'UNET':
        from ap_architectures.models import UNET as NET
    elif prms_net['arch'] == 'CNET':
        from ap_architectures.models import CNET as NET
    
    tf_config.run_functions_eagerly(debugger_is_active())

    fcn_train(prms, prms_net, NET)

