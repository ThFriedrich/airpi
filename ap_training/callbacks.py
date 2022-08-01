import gc
from matplotlib import pyplot as plt
from tensorflow import summary, keras, math, image, complex64, cast as tf_cast, unstack
from os import path

from tqdm import tqdm
from ap_reconstruction import reconstruction_functions as rf
from ap_reconstruction.airpi_dataset import airpi_dataset

from numpy import cast, ceil, angle

from ap_training.data_fcns import getTestDataset
from ap_training.lr_scheduler import lr_schedule
from ap_utils.util_fcns import PRM, fcn_plot_example, save_hparams, fcn_plot_nn_in_out


class airpi_callbacks:
    def __init__(self, prms_o, prms_net, tb_freq, im_freq, reset_metric_freq_batch):
        str_lr = '%.2E' % prms_o['learning_rate_0']
        log_sub_dir = path.join(
            prms_o['log_path'], 'bs_' + str(prms_o['batch_size']) + '_lr_' + str_lr)

        writer = summary.create_file_writer(
            logdir=path.join(log_sub_dir, 'images'))
        writer.set_as_default()

        callback_cp = self.Checkpoint(
            log_dir=prms_o['cp_path'] + '/cp-best.ckpt',
            prms=prms_o,
            prms_net=prms_net)
        callback_im = self.TensorboardImage(
            im_freq=im_freq,
            test_ds=prms_o['test_path'],
            sample_dir=prms_o['sample_dir'],
            reset_metric_freq=reset_metric_freq_batch)
        callback_tb = keras.callbacks.TensorBoard(
            log_dir=log_sub_dir,
            write_graph=True,
            profile_batch=1,
            histogram_freq=1,
            update_freq='epoch')
        # callback_lr = self.LearningRateScheduler(
        #     log_dir=log_sub_dir,
        #     prms=prms_o,
        #     prms_net=prms_net)
        callback_lr = self.LearningRateScheduler2(
            prms_net=prms_net,
            prms=prms_o,)
        callback_csv = keras.callbacks.CSVLogger(
            path.join(log_sub_dir, 'log.csv'),
            separator=';',
            append=True)

        self.as_list = [callback_cp, callback_im,
                        callback_tb, callback_lr, callback_csv]

    class TensorboardImage(keras.callbacks.Callback):
        def __init__(self, reset_metric_freq, im_freq, test_ds, sample_dir):
            self.reset_metric_freq = reset_metric_freq
            self.im_freq = im_freq
            self.n_dat = 16
            self.ds = getTestDataset(test_ds, self.n_dat)
            self.sample_dir = sample_dir
            self.rec_prms = [{
                "name":"Graphene",
                "path": self.sample_dir+"graphene/gra.hdf5",
                # "path": "/home/thomas/SSD/Samples/graphene/gra.hdf5",
                "key":"ds",
                "dose":None,
                "E0": 200.0,
                "apeture": 25,
                "gmax": 2.5,
                "cbed_size": 128,
                "step_size": 0.2,
                "aberrations": [-1, 0.001],
                "bfm_type": 'avrg',
                "oversample": 2.0,
                "step":1,
                "options":{'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}
            }, {
                "name":"MoS2",
                "path": self.sample_dir+"MSO/airpi_sto.h5",
                "key":"ds_int",
                "dose":500,
                "E0": 300.0,
                "apeture": 20.0,
                "gmax": 4.5714,
                "cbed_size": 128,
                "step_size": 0.05,
                "aberrations": [-1, 1e-3],
                "bfm_type": 'avrg',
                "oversample": 2.0,
                "step":4,
                "options":{'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}
            }, {
                "name":"STO",
                "path": self.sample_dir+"STO/hole_preprocessed_cropped_2.h5",
                "key":"ds",
                "dose":None,
                "E0": 300.0,
                "apeture": 20.0,
                "gmax": 1.6671,
                "cbed_size": 64,
                "step_size": 0.1818,
                "aberrations": [-1, 1e-3],
                "bfm_type": 'avrg',
                "oversample": 2.0,
                "step":1,
                "options":{'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}
            }]

            for prm in self.rec_prms:
                prm['step_size'] *= prm['step']


        def deploy_out(self, x):
            x_a, x_p = unstack(x, axis=-1)
            x_a = tf_cast(x_a**5, complex64)
            x_p = tf_cast(x_p, complex64)
            x_o = x_a * math.exp(1j*x_p)
            return x_o

        def run_example_ds(self,epoch):
            for prm in self.rec_prms:
                try:
                    example_ds = airpi_dataset(
                        prm, prm['path'], prm['key'], prm['dose'], step=prm['step'], in_memory=False)
                    example_ds.ds = example_ds.ds.batch(
                        64, drop_remainder=False).prefetch(8)
                    steps = cast['int'](ceil(example_ds.ds_n_dat/64))
                    t = tqdm(unit=' samples', total=example_ds.ds_n_dat)
                    worker = rf.ReconstructionWorker(
                        64, example_ds.rec_prms, options=prm['options'])
                    ds_iter = iter(example_ds.ds)
                    for _ in range(steps):
                        set = next(ds_iter)
                        # set['cbeds'] = image.resize(set['cbeds'], [64, 64])
                        pred = self.model.predict_on_batch(set['cbeds'])
                        pred = self.deploy_out(pred)
                        worker.update_patch(pred, set['pos'])
                        t.update(pred.shape[0])
                    # worker.ThreadPool.shutdown(True)
                    le_half = int(worker.cbed_size_scaled//2)
                    obj = angle(
                        worker.object[le_half:-le_half, le_half:-le_half])
                    t.close()
                    im = fcn_plot_example(obj, epoch, prm["name"])
                    summary.image(name=prm["name"], data=im,
                              step=epoch, max_outputs=256)
                    del(worker, ds_iter, set, pred, t, im, obj, le_half, steps, example_ds)
                except:
                    pass
                gc.collect()

        def fcn_xy_image_gen(self, pred, epoch):
            '''Create Plots to write images to Tensorboard'''

            for x, y in self.ds.take(1):
                feat = x.numpy()
                lab = y.numpy()

            for ix in range(self.n_dat):
                im = fcn_plot_nn_in_out(
                    feat[ix, ...], pred[ix, ...], lab[ix, ...], epoch, str(ix))
                summary.image(name='test_set_'+str(ix), data=im,
                              step=epoch, max_outputs=256)

        def on_epoch_end(self, epoch, logs=None):
            '''Scheduleing'''
            if epoch % self.im_freq == 0:
                pred = self.model.predict(self.ds, steps=1)
                self.fcn_xy_image_gen(pred, epoch)
                self.run_example_ds(epoch)

    class Checkpoint(keras.callbacks.ModelCheckpoint):
        def __init__(self, log_dir, prms, prms_net):
            super(airpi_callbacks.Checkpoint, self).__init__(log_dir)

            self.prms = prms
            self.prms_net = prms_net
            self.filepath = log_dir
            self.verbose = 1
            self.monitor = 'val_loss'
            self.mode = 'min'
            self.save_best_only = True
            self.save_weights_only = True
            self.hp_path = path.join(
                self.prms['cp_path'], 'hyperparameters.pickle')

        def on_epoch_end(self, epoch, logs=None):
            self.prms['learning_rate'] = float(
                keras.backend.get_value(self.model.optimizer.lr))
            save_hparams([self.prms, self.prms_net], self.hp_path)
            return super().on_epoch_end(epoch, logs=logs)

        def on_epoch_begin(self, epoch, logs=None):
            self.prms['ep'] = epoch
            return super().on_epoch_begin(epoch, logs=logs)

    class LearningRateScheduler(keras.callbacks.Callback):
        """Custom learning rate schedule callback """

        def __init__(self, log_dir, prms, prms_net):
            super(airpi_callbacks.LearningRateScheduler, self).__init__()
            self.prms = prms
            self.prms_net = prms_net
            self.schedule = lr_schedule(prms).schedule

        def on_epoch_begin(self, epoch, logs=None):
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            # Get the current learning rate from model's optimizer.
            lr = float(keras.backend.get_value(self.model.optimizer.lr))
            scheduled_lr = self.schedule[epoch]

            # For grid search a range a lr values is used.
            if 'learning_rate_rng' in self.prms:
                self.model.load_weights(path.join(
                    path.curdir, self.prms['cp_path'], 'initial_weights.h5'))

            if abs(scheduled_lr-lr) > 1e-8:
                # Set the value back to the optimizer before this epoch starts
                keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
                print('\nEpoch %05d: Learning rate changed to %6.5e.' %
                      (epoch + 1, scheduled_lr))

            # Tensorboard metrics
            summary.scalar('Learning Rate',
                           data=self.model.optimizer.lr, step=epoch)
            summary.scalar(
                'Batch Size', data=self.prms['batch_size'], step=epoch)
            summary.scalar('Dose_l', data=PRM.dose[0], step=epoch)
            summary.scalar('Dose_u', data=PRM.dose[-1], step=epoch)
            if self.prms_net['w_regul'] != None:
                summary.scalar('Weight Regularisation',
                               data=self.prms_net['w_regul'], step=epoch)
            if self.prms_net['a_regul'] != None:
                summary.scalar('Activity Regularisation',
                               data=self.prms_net['a_regul'], step=epoch)
            return super().on_epoch_begin(epoch, logs=logs)
    class LearningRateScheduler2(keras.callbacks.ReduceLROnPlateau):
        """Custom learning rate schedule callback """

        def __init__(self, prms, prms_net):
            super(airpi_callbacks.LearningRateScheduler2, self).__init__(
                monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            mode='min',
            min_delta=1e-8,
            cooldown=0,
            warmup=0,
            min_lr=1e-8)
            self.prms_net = prms_net
            self.prms = prms
            

        def on_epoch_begin(self, epoch, logs=None):
            # Tensorboard metrics
            summary.scalar('Learning Rate',
                           data=self.model.optimizer.lr, step=epoch)
            summary.scalar(
                'Batch Size', data=self.prms['batch_size'], step=epoch)
            summary.scalar('Dose_l', data=PRM.dose[0], step=epoch)
            summary.scalar('Dose_u', data=PRM.dose[-1], step=epoch)
            if self.prms_net['w_regul'] != None:
                summary.scalar('Weight Regularisation',
                               data=self.prms_net['w_regul'], step=epoch)
            if self.prms_net['a_regul'] != None:
                summary.scalar('Activity Regularisation',
                               data=self.prms_net['a_regul'], step=epoch)
            return super().on_epoch_begin(epoch, logs=logs)
