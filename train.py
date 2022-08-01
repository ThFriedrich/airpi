"""Module for Tensorflow Training"""
import os
import argparse
import silence_tensorflow.auto

def fcn_train(prms_o, prms_net, NET):
    '''Train the Network'''

    training_ds, validation_ds = getDatasets(prms_o)

    model = NET(prms_net, training_ds._flat_shapes[0][1:4]).build()
    model.summary()

    tb_freq = 4
    if os.path.isfile(os.path.join(prms_o['cp_path'], 'checkpoint')) is True:
        model.load_weights(prms_o['cp_path'] + '/cp-best.ckpt')
        print('Resuming training from Epoch ' + str(prms['ep']))

    # loss_fcn = ls.loss(prms_o)
    # metric_fcn = ls.metric_fcns(prms_o, model)
    loss_fcn = ls.loss
    metric_fcns = ls.metrics()
    model.compile(
        optimizer=optimizers.Adam(prms_o['learning_rate']),loss=loss_fcn, metrics=metric_fcns)
        # optimizer=optimizers.SGD(prms_o['learning_rate']), loss=loss_fcn, metrics=metric_fcns)
        # optimizer=optimizers.Nadam(prms_o['learning_rate']),loss=loss_fcn, metrics=[metric_fcn])
        # optimizer=optimizers.Adam(prms_o['learning_rate']),loss=loss_fcn)
        # optimizer=optimizers.Adam(prms_o['learning_rate']),loss=loss_fcn, metrics=[metric_fcn])
       
    
    callbacks = airpi_callbacks(prms_o, prms_net, tb_freq, 1, tb_freq).as_list

    from tensorflow.data.experimental import AUTOTUNE

    model.fit(training_ds,
              validation_data=validation_ds,
              validation_steps=prms_o['validation_steps'],
              validation_freq=1,
              epochs=prms_o['epochs'],
              steps_per_epoch=prms_o['steps_per_epoch'],
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
    args = vars(parser.parse_args())

    if args["gpu_id"] is None:
        args["gpu_id"] = 1
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])


    from tensorflow import config as tf_config

    # physical_devices = tf_config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf_config.experimental.set_memory_growth(physical_devices[0], True)
   
    from ap_utils.util_fcns import debugger_is_active, manage_args
    prms, prms_net = manage_args(args) 

    from tensorflow.keras import mixed_precision, optimizers
    
    # mixed_precision.set_global_policy('mixed_float16')
    # mixed_precision.set_global_policy('float32')
    tf_config.optimizer.set_jit("autoclustering")
   

    from ap_training.data_fcns import getDatasets
    from ap_training.callbacks import airpi_callbacks
    if prms_net['arch'] == 'UNET':
        from ap_architectures.models import UNET as NET
    elif prms_net['arch'] == 'COMPLEX':
        from ap_architectures.complex_net import CNET as NET
    import ap_training.losses as ls
    
    tf_config.run_functions_eagerly(debugger_is_active())

    fcn_train(prms, prms_net, NET)

