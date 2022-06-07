from tensorflow import summary, keras, math
from os import path

from ap_training.data_fcns import getTestDataset
from ap_training.lr_scheduler import lr_schedule
from ap_utils.util_fcns import PRM, save_hparams, fcn_plot_nn_in_out

class airpi_callbacks:
    def __init__(self, prms_o, prms_net, tb_freq, im_freq, reset_metric_freq_batch): 
        str_lr = '%.2E' % prms_o['learning_rate_0']
        log_sub_dir = path.join(prms_o['log_path'], 'bs_' + str(prms_o['batch_size']) + '_lr_' + str_lr)

        callback_cp = self.Checkpoint(
            log_dir=prms_o['cp_path'] + '/cp-best.ckpt',
            prms=prms_o,
            prms_net=prms_net)
        callback_im = self.TensorboardImage(
            im_freq=im_freq,
            test_ds=prms_o['test_path'],
            reset_metric_freq = reset_metric_freq_batch)
        callback_tb = keras.callbacks.TensorBoard(
            log_dir=log_sub_dir,
            write_graph=True,
            update_freq=tb_freq)
        callback_lr = self.LearningRateScheduler(
            log_dir=log_sub_dir,
            prms=prms_o,
            prms_net=prms_net)
        callback_csv = keras.callbacks.CSVLogger(
            path.join(log_sub_dir,'log.csv'), 
            separator=';', 
            append=True)

        writer = summary.create_file_writer(logdir=log_sub_dir+'/images')
        writer.set_as_default()

        self.as_list = [callback_cp, callback_im, callback_tb, callback_lr, callback_csv]

    class TensorboardImage(keras.callbacks.Callback):
        def __init__(self, reset_metric_freq, im_freq, test_ds):
            self.reset_metric_freq = reset_metric_freq
            self.im_freq = im_freq
            self.n_dat = 8
            self.ds = getTestDataset(test_ds, self.n_dat)
            
        def fcn_xy_image_gen(self, pred, epoch):
            '''Create Plots to write images to Tensorboard'''

            for x, y in self.ds.take(1):
                feat = x.numpy()
                lab = y.numpy()

            for ix in range(self.n_dat):
                im = fcn_plot_nn_in_out(feat[ix,...], pred[ix,...], lab[ix,...], epoch, str(ix))
                summary.image(name='test_set_'+str(ix), data=im, step=epoch,max_outputs=256)

        def on_epoch_end(self, epoch, logs=None):
            '''Scheduleing'''
            if epoch % self.im_freq == 0:
                pred = self.model.predict(self.ds, steps = 1)
                self.fcn_xy_image_gen(pred, epoch)

        def on_train_batch_begin(self, batch, logs=None):
            if batch > 0 and batch % self.reset_metric_freq == 0:
                self.model.reset_metrics()


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
            self.hp_path = path.join(self.prms['cp_path'], 'hyperparameters.pickle')

        def on_epoch_end(self, epoch, logs=None):     
            self.prms['learning_rate'] = float(keras.backend.get_value(self.model.optimizer.lr))
            save_hparams([self.prms, self.prms_net], self.hp_path)
            return super().on_epoch_end(epoch, logs=logs)
        
        def on_epoch_begin(self, epoch, logs=None):
            self.prms['ep'] = epoch
            return super().on_epoch_begin(epoch, logs=logs)

    class LearningRateScheduler(keras.callbacks.Callback):
        """Custom learning rate schedule callback """

        def __init__(self, log_dir, prms, prms_net):
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
            summary.scalar('Learning Rate', data=self.model.optimizer.lr, step=epoch)
            summary.scalar('Batch Size', data=self.prms['batch_size'], step=epoch)
            summary.scalar('Dose_l', data=PRM.dose[0], step=epoch)
            summary.scalar('Dose_u', data=PRM.dose[-1], step=epoch)
            if self.prms_net['w_regul'] != None:
                summary.scalar('Weight Regularisation',
                            data=self.prms_net['w_regul'], step=epoch)
            if self.prms_net['a_regul'] != None:
                summary.scalar('Activity Regularisation',
                            data=self.prms_net['a_regul'], step=epoch)
            
            return super().on_epoch_begin(epoch, logs=logs)
