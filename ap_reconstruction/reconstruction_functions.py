import numpy as np
from math import pi
import os
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from tqdm import tqdm
import nvidia_smi
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from ap_reconstruction.colormaps import parula

def get_gpu_memory():

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    nvidia_smi.nvmlShutdown()

    return info

# gpus = tf.config.list_physical_devices('GPU')
# gpu_mem = get_gpu_memory().free/1024**2
# tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem-512)])

from ap_architectures.utils import fft2d, ifft2d, interp2dcomplex, tf_com, tf_ifft2d, rAng_2_mrad, tf_normalise_to_one, tf_normalise_to_one_complex, tf_pi, tf_fft2d, tf_FourierShift2D

def load_model(prms_net, X_SHAPE, scale):
    '''
    Load the UNET with given parameters and Checkpoint
    '''

    if 'CNET' in prms_net['cp_path'].upper():
        from ap_architectures.complex_net import CNET as ARCH
    else:
        from ap_architectures.models import UNET as ARCH

    if os.path.isfile(os.path.join(prms_net['cp_path'], 'checkpoint')):
        print('Use checkpoint: ' + str(prms_net['cp_path']))
        model = ARCH(prms_net, X_SHAPE, deploy=True).build()
        model.load_weights(prms_net['cp_path']+'/cp-best.ckpt')
        model.summary()
        return model
    else:
        warnings.warn("Warning! No Checkpoint found! The loaded model is untrained!")
     

class ReconstructionWorker():
    """Custom callback to write NN-output to disk asynchronously"""
    def __init__(self, cbed_size, rec_prms, options={'b_offset_correction':False}):
        self.ds_ews = options['ew_ds_path']
        self.rec_prms = rec_prms
        # self.cbed_size_scaled = np.floor(cbed_size*self.rec_prms['oversample'])
        # if self.cbed_size_scaled % 2 != 0: self.cbed_size_scaled += 1

        self.b_offset_correction = options['b_offset_correction']

        if self.ds_ews is not None:
            self.hf_write = h5py.File(ds_ews, 'w')
            self.ds_write =     self.hf_write.create_dataset('ds', 
                                shape=(1, 1, int(cbed_size), int(cbed_size)), 
                                chunks = (1,1,int(cbed_size),int(cbed_size)), 
                                maxshape=(None, None, int(cbed_size), int(cbed_size)), 
                                dtype='complex64')
        self.create_obj(cbed_size)  
        if options['threads'] > 1:
            self.__multithreading__ = True
            self.ThreadPool = ThreadPoolExecutor(1)
            self.__lock__ = Lock()
        else:
            self.__multithreading__ = False

    def create_obj(self, cbed_size):
        self.rec_prms["space_size"] = cbed_size/(2*self.rec_prms["gmax"])
        self.rec_prms["scale"] = self.rec_prms["oversample"]*self.rec_prms["space_size"]/cbed_size/self.rec_prms["step_size"]
        
        if self.rec_prms["scale"] > 1:
            self.cbed_size_scaled = int(np.ceil(cbed_size * self.rec_prms["scale"]))
        else:
            self.cbed_size_scaled = int(cbed_size)

        self.rec_prms["px_size"] = self.rec_prms["space_size"]/self.cbed_size_scaled
    
        self.rg = np.arange(int(self.cbed_size_scaled))*2.0*np.pi*1j/self.cbed_size_scaled
        self.st_px = self.rec_prms['step_size']/self.rec_prms['px_size']
        self.t = np.zeros((int(self.cbed_size_scaled), int(self.cbed_size_scaled)))
        self.cbed_center = int(np.ceil((self.cbed_size_scaled+1)/2))-1

        self.le_y = int(round(self.rec_prms['ny']*self.rec_prms["step_size"]/self.rec_prms["px_size"])+self.cbed_size_scaled-1)
        self.le_x = int(round(self.rec_prms['nx']*self.rec_prms["step_size"]/self.rec_prms["px_size"])+self.cbed_size_scaled-1)
        self.object = np.zeros((self.le_x, self.le_y), dtype='complex64')
        
        beam_k = self.rec_prms['beam_in_k'][...,0]*np.exp(1j*self.rec_prms['beam_in_k'][...,1])
        beam_k = interp2dcomplex(beam_k,out_size=[64,64])
        beam_k = tf_normalise_to_one_complex(beam_k)
        beam_k = self.pad_cbed(beam_k)

        beam_r = fft2d(beam_k)/self.cbed_size_scaled

        self.rec_prms['beam_in_r'] = tf.cast(beam_r,tf.complex64)
        self.rec_prms['beam_in_k'] = tf.cast(beam_k,tf.complex64)
        
        beam_r_int = np.abs(beam_r)**2
        self.idx_b = np.nonzero(beam_r_int>0.01*(np.max(beam_r_int))) # where the beam has significant intensity

        self.t[self.idx_b] = 1
        self.idx_bb = self.translate_idx()
        weight = beam_r_int
        weight = weight[self.idx_b]
        self.weight = weight / np.sum(weight)

    def pad_cbed(self, cbed_in):
        nx_in = np.size(cbed_in,0)
        nx_out = int(self.cbed_size_scaled)
        if nx_out/nx_in > 1:
            center_in = self.find_center(nx_in)
            center_out = self.find_center(nx_out)
            t = int(center_out-center_in)
            cbed_out = np.zeros((nx_out,nx_out), dtype='complex64')
            cbed_out[t:t+nx_in,t:t+nx_in] = cbed_in
        else:
            cbed_out = cbed_in
        return cbed_out

    def find_center(self, nx):
        return np.ceil((nx + 1) / 2)
    
    def locate(self, x, y):
        xx = (x-1.0)*self.st_px + 1.0
        yy = (y-1.0)*self.st_px + 1.0
        x_int = np.round(xx)
        x_frac = xx-x_int
        y_int = np.round(yy)
        y_frac = yy-y_int
        return x_int, x_frac, y_int, y_frac

    def translate_idx(self):
        tt = np.zeros((self.le_x, self.le_y))
        tt[0:int(self.cbed_size_scaled),0:int(self.cbed_size_scaled)] = self.t
        return np.nonzero(tt==1)

    def fcn_beam_shift_px(self, y, x):
        xlin = np.exp(self.rg*y)
        ylin = np.exp(self.rg*x)
        Y,X = np.meshgrid(ylin,xlin)
        phase = X*Y
        return phase/phase[self.cbed_center, self.cbed_center]

    def build_obj(self, beam_out, shift_y, shift_x):
        phase_ramp = self.fcn_beam_shift_px(shift_y, shift_x)
        beam_in = self.rec_prms['beam_in_k'] * phase_ramp
        # beam_in_s = tf.squeeze(tf_FourierShift2D(beam_in[tf.newaxis,...], tf.constant([0.5, 0.5], dtype=tf.float32,shape=[1,2])))
        # beam_out_s = tf.squeeze(tf_FourierShift2D(beam_out[tf.newaxis,...], tf.constant([0.5, 0.5], dtype=tf.float32,shape=[1,2])))
        beam_out_s = beam_out * phase_ramp
        beam_in_r = fft2d(beam_in)/self.cbed_size_scaled
        beam_out_r = fft2d(beam_out_s)/self.cbed_size_scaled
        # self.plot_output(beam_out_r, beam_out_r / beam_in_r)
        return beam_out_r, beam_in_r

    def update(self, obj, coor):
        idx_bb = (self.idx_bb[0]+int(coor[0]-1),self.idx_bb[1]+int(coor[1])-1)
        phase = np.angle(obj)
        if self.b_offset_correction:
            if np.all(coor == [1,1]):
                offset = 0
            else:
                phase_big = np.angle(self.object[idx_bb])
                offset = np.sum((phase-phase_big)*self.weight)
            obj = np.exp(1j*(phase-offset*0.6))*self.weight
        else:
            obj = np.exp(1j*phase)*self.weight
        
        if self.__multithreading__:
            with self.__lock__:
                self.object[idx_bb] += obj
        else:
            self.object[idx_bb] += obj


    def update_patch(self, data, yx_ov=None):
        if yx_ov is not None:
            data_c = []
            data_c.append(data)
            data_c.append(yx_ov)
            data = data_c

        for b in range(data[0].shape[0]):
            beam_out = tf.squeeze(data[0][b,...])
            yx = np.cast['float'](data[1][b,...])
            
            x_int, x_frac, y_int, y_frac = self.locate(yx[1],yx[0])  
            beam_out = self.pad_cbed(beam_out)
            beam_out, beam_in = self.build_obj(beam_out, y_frac, x_frac)
            obj = beam_out[self.idx_b] / beam_in[self.idx_b]
            self.update(obj, [y_int, x_int])

            if self.ds_ews is not None:
                self.ds_write_to_disk(data,b)

            # with self.__lock__:
            #     self.plot_output(beam_out, beam_out / beam_in)
            #     self.plot_output(tf.squeeze(data[0][b,...]), beam_out / beam_in)
                # self.object[start_xy[0]:start_xy[0]+self.cbed_size_scaled, start_xy[1]:start_xy[1]+self.cbed_size_scaled] += obj

            
    def ds_write_to_disk(self, data, b):
        pos = data[1][b,...] 
        pred = np.squeeze(data[0][b,...])
        #Extend Dataset and write
        if self.ds_write.shape[0] < pos[0]:
            self.ds_write.resize(pos[0],axis=0)
        if self.ds_write.shape[1] < pos[1]:
            self.ds_write.resize(pos[1],axis=1)
        self.ds_write[pos[0]-1,pos[1]-1,...] = pred
        self.ds_write.flush()
        

    def plot_output(self, pred, obj):


        cols = ['Predicted Amplitude','Predicted Phase','Beam Amplitude','Beam Phase','Object Amplitude','Object Phase']
        font = {'fontsize': 10, 'fontweight' : 1, 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}

        fig , axes = plt.subplots(3, 2,figsize=(4, 4),squeeze=True)

        # pred = (pred[...,0]**5)*np.exp(1j*pred[...,1])
        im = list()

        im.append(axes[0,0].imshow(np.abs(pred)**0.2))
        im.append(axes[0,1].imshow(np.angle(pred)))
        im.append(axes[1,0].imshow(np.abs(self.rec_prms['beam_in_r'])))
        im.append(axes[1,1].imshow(np.angle(self.rec_prms['beam_in_r'])))
        im.append(axes[2,0].imshow(np.abs(obj)))
        im.append(axes[2,1].imshow(np.angle(obj)))

        for ix, axi in enumerate(axes.ravel()):
            axi.set_axis_off()
            axi.set_title(cols[ix],fontdict=font)
                
        plt.tight_layout()
        plt.savefig('prediction_a.png')
        plt.close(fig)

def update_obj_fig(writer, obj_fig, fig, xy):
    le_half = int(writer.cbed_size_scaled//2)
    st_px = writer.rec_prms['step_size']/writer.rec_prms['px_size']
    # data = np.angle(writer.object)
    data = np.angle(writer.object[le_half:-le_half,le_half:-le_half])
    ylim = int(st_px*np.max(xy[:,0]))
    data[ylim:,:] = np.core.nan
    obj_fig.set_data(data)
    obj_fig.set_clim(np.nanmin(data), np.nanmax(data))
    fig.canvas.flush_events() 

def plot_set_init(writer):
    le_half = int(writer.cbed_size_scaled//2)
    set_fig, set_ax = plt.subplots(3,5)
    set_ax_obj = []
    for iy in range(3):
        for ix in range(3):
            set_ax_obj.append(set_ax[iy, ix].imshow(np.ones((128,128))))
    
    set_ax_obj.append(set_ax[0, 3].imshow(np.ones((128,128))))
    set_ax_obj.append(set_ax[0, 4].imshow(np.ones((128,128))))
    set_ax_obj.append(set_ax[1, 3].imshow(np.ones((128,128))))
    set_ax_obj.append(set_ax[1, 4].imshow(np.ones((128,128))))

    set_ax_obj.append(set_ax[2, 3].imshow(np.angle(writer.object[le_half:-le_half,le_half:-le_half]),vmin=-pi, vmax=pi))
    set_ax_obj.append(set_ax[2, 4].imshow(np.ones((128,128))))
    [axi.set_axis_off() for axi in set_ax.ravel()]
    set_fig.tight_layout()
    return set_fig, set_ax_obj

def plot_set_update(set_ax_obj, set_fig, set, pred, pos, writer):
        order = [6, 3, 0, 7, 4, 1, 8, 5, 2]
        le_half = int(writer.cbed_size_scaled//2)
        st_px = writer.rec_prms['step_size']/writer.rec_prms['px_size']
        for ib, set_b in enumerate(set['cbeds']):
            if (ib+1) == len(set['cbeds']):
                for ix, ix_s in enumerate(order):
                    data = set_b[...,ix]**0.1
                    set_ax_obj[ix_s].set_data(data)
                    set_ax_obj[ix_s].set_clim(np.min(data), np.max(data))
            
                pred_b = np.squeeze(pred[ib,...])
                pred_b = writer.pad_cbed(pred_b)
                pred_br = ifft2d(pred_b)*writer.cbed_size_scaled

                set_ax_obj[9].set_data(np.angle(pred_b))
                set_ax_obj[9].set_clim(-pi, pi)
                set_ax_obj[10].set_data(np.abs(pred_b))
                set_ax_obj[10].set_clim(np.min(np.abs(pred_b)), np.max(np.abs(pred_b)))

                set_ax_obj[11].set_data(np.angle(pred_br))
                set_ax_obj[11].set_clim(-pi, pi)
                aa = np.abs(pred_br)
                set_ax_obj[12].set_data(aa)
                set_ax_obj[12].set_clim(np.min(aa), np.max(aa))

                data = np.angle(writer.object[le_half:-le_half,le_half:-le_half])
                ylim = int(st_px*np.max(pos[:,0]))
                data[ylim-1:,:] = np.core.nan
                set_ax_obj[13].set_data(data)
                set_ax_obj[13].set_clim(np.nanmin(data), np.nanmax(data))

                beam = writer.rec_prms['beam_in_r']
                obj = np.angle(pred_br/beam)
                msk = np.zeros_like(obj)
                msk[writer.idx_b] = 1.0

                set_ax_obj[14].set_data(obj*msk)
                set_ax_obj[14].set_clim(-np.pi, np.pi)
                set_fig.canvas.flush_events() 
    # plt.savefig('set.png')

def retrieve_phase_from_generator(ds_class, prms_net, options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}):
    plt.ion()
    # bs = int(pow(2, np.floor(np.log(gpu_mem/60/scale)/np.log(2))))
    bs = 256
    steps= np.cast[np.int32](np.ceil(ds_class.ds_n_dat/bs))
    writer = ReconstructionWorker(64, ds_class.rec_prms, options)
    # ds_class.get_bfm('avrg')

    model = load_model(prms_net, ds_class.ds._flat_shapes[0], writer.cbed_size_scaled)
    ds_class.ds = ds_class.ds.batch(bs, drop_remainder=False).prefetch(8)
    ds_iter = iter(ds_class.ds)
    
    fig, ax = plt.subplots()
    le_half = int(writer.cbed_size_scaled//2)
    obj_fig = ax.imshow(np.angle(writer.object[le_half:-le_half,le_half:-le_half]),vmin=-pi, vmax=pi, cmap=parula)
    set_fig, set_ax = plot_set_init(writer)
    t= tqdm(unit=' samples', total=ds_class.ds_n_dat)
    for _ in range(steps):
        # try:
        set = next(ds_iter)
        pred = model.predict_on_batch(set['cbeds'])
        if writer.__multithreading__:
            writer.ThreadPool.submit(writer.update_patch, pred, set['pos'])
        else:
            writer.update_patch(pred, set['pos']) 
        update_obj_fig(writer, obj_fig, fig, set['pos'])
        plot_set_update(set_ax, set_fig, set, pred, set['pos'], writer)
        t.update(pred.shape[0])
        # except StopIteration:
        #     print('Generator ran out of Data!')
    if writer.__multithreading__:
        writer.ThreadPool.shutdown(True)
    t.close()
    update_obj_fig(writer, obj_fig, fig, set['pos'])
    # plt.savefig('reconstruction7.png')
    plt.colorbar(obj_fig,ax=ax)
    plt.show(block=True)
    