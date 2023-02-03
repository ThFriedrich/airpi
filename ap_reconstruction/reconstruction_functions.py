import time
import numpy as np
from math import ceil, floor, pi
import os
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from ap_utils.colormaps import parula
from ap_utils.functions import tf_cast_complex, tf_complex_interpolation, tf_fft2d, tf_ifft2d, tf_normalise_to_one_complex, tf_pad2d, tf_probe_function

def load_model(prms_net, X_SHAPE, bs, probe):
    '''
    Load the UNET with given parameters and Checkpoint
    '''

    if 'C_' in prms_net['cp_path'].upper():
        from ap_architectures.models import CNET as ARCH
    else:
        from ap_architectures.models import UNET as ARCH

    if os.path.isfile(prms_net['cp_path']) and prms_net['cp_path'].endswith('tflite'):
        from ap_architectures.models import TFLITE_Model
        model = TFLITE_Model(model_path=prms_net['cp_path']+'.tflite')

    if os.path.isfile(os.path.join(prms_net['cp_path'], 'checkpoint')):
        print('Use checkpoint: ' + str(prms_net['cp_path']))
        model = ARCH(prms_net, X_SHAPE, deploy=True).build(bs)
        # model.build(probe,bs)
        model.load_weights(prms_net['cp_path']+'/cp-best.ckpt')
        # model.summary()
        return model
    else:
        warnings.warn("Warning! No Checkpoint found! The loaded model is untrained!")
     

class ReconstructionWorker():
    """Custom callback to write NN-output to disk asynchronously"""
    def __init__(self, cbed_size, rec_prms, options={'b_offset_correction':False}):
        self.ds_ews = options['ew_ds_path']
        self.rec_prms = rec_prms
        self.y_pos = 0

        self.b_offset_correction = options['b_offset_correction']

        if self.ds_ews is not None:
            self.hf_write = h5py.File(self.ds_ews, 'w')
            self.ds_write =     self.hf_write.create_dataset('ds', 
                                shape=(1, 1, int(cbed_size), int(cbed_size)), 
                                chunks = (1,1,int(cbed_size),int(cbed_size)), 
                                maxshape=(None, None, int(cbed_size), int(cbed_size)), 
                                dtype='complex64')
        self.create_obj(cbed_size)  
        self.create_phase_ramp_lib(cbed_size)
        self.__lock__ = Lock()
        if options['threads'] > 1:
            self.__multithreading__ = True
            self.ThreadPool = ThreadPoolExecutor(options['threads'])
        else:
            self.__multithreading__ = False

    @tf.function(jit_compile=True)
    def make_phase_ramp_lib(self, cbed_size):
        self.phase_ramp_lib = tf.zeros((2,10,cbed_size,cbed_size))
        for shift in range(10):
            self.phase_ramp_lib[0,shift] = \
                self.fcn_beam_shift_px(shift, 0) # shift y
            self.phase_ramp_lib[1,shift] = \
                self.fcn_beam_shift_px(0, shift-0.5) # shift x

    def create_obj(self, cbed_size):
        self.rec_prms["space_size"] = cbed_size/(2*self.rec_prms["gmax"])
        self.rec_prms["scale"] = self.rec_prms["oversample"]*self.rec_prms["space_size"]/cbed_size/self.rec_prms["step_size"]

        if self.rec_prms["scale"] > 1:
            self.cbed_size_scaled = int(ceil(cbed_size * self.rec_prms["scale"]))
        else:
            self.cbed_size_scaled = int(cbed_size)

        self.rec_prms["px_size"] = self.rec_prms["space_size"]/self.cbed_size_scaled
        gmax_s = tf.cast(self.cbed_size_scaled/(2*self.rec_prms["space_size"]),tf.float32)

        self.rg = tf.cast(np.arange(int(self.cbed_size_scaled))*2.0*np.pi*1j/self.cbed_size_scaled, tf.complex64)
        self.st_px = tf.cast(self.rec_prms['step_size']/self.rec_prms['px_size'],tf.float32)
        self.t = np.zeros((int(self.cbed_size_scaled), int(self.cbed_size_scaled)))
        self.cbed_center = int(np.ceil((self.cbed_size_scaled+1)/2))-1

        self.le_y = int(round(self.rec_prms['ny']*self.st_px.numpy())+self.cbed_size_scaled-1)
        self.le_x = int(round(self.rec_prms['nx']*self.st_px.numpy())+self.cbed_size_scaled-1)
        self.object = np.zeros((self.le_y, self.le_x), dtype='complex64')
        self.le_half = int(self.cbed_size_scaled//2)

        self.beam_k = self.pad_cbed(self.rec_prms['beam_in_k'])
        self.beam_r = tf_cast_complex(self.rec_prms['beam_in_r'][...,0], self.rec_prms['beam_in_r'][...,1])
        self.beam_r_scaled = tf_probe_function(self.rec_prms['E0'], self.rec_prms['apeture'], gmax_s,
                                          self.cbed_size_scaled, self.rec_prms['aberrations'], domain='r', type='complex', refine=True)
        self.beam_r = tf_normalise_to_one_complex(self.beam_r)

        
        beam_r_int = np.abs(self.beam_r)**2
        beam_r_sc_int = np.abs(self.beam_r_scaled)**2
        # where the beam has significant intensity
        self.idx_b = np.nonzero(beam_r_int>0.1*(np.max(beam_r_int))) 
        self.idx_b_sc = np.nonzero(beam_r_sc_int>0.1*(np.max(beam_r_sc_int))) 
        self.idx_b_sc_tf = tf.transpose(tf.stack((self.idx_b_sc[0],self.idx_b_sc[1])))
        weight = beam_r_int
        self.weight = tf.cast(weight / np.sum(weight),tf.complex64)
        
        self.t[self.idx_b_sc] = 1
        self.idx_bb = self.translate_idx()


    def pad_cbed(self, cbed_in):
        nx_in = np.size(cbed_in,0)
        nx_out = int(self.cbed_size_scaled)
        if nx_out/nx_in > 1:
            center_in = self.find_center(nx_in)
            center_out = self.find_center(nx_out)
            t = int(center_out-center_in)
            cbed_out = np.zeros((nx_out, nx_out), dtype='complex64')
            cbed_out[t:t+nx_in,t:t+nx_in] = cbed_in
        else:
            cbed_out = cbed_in
        return cbed_out

    @tf.function(jit_compile=True)
    def tf_pad_cbed(self, cbed_in):
        nx_in = int(cbed_in.shape[1])
        nx_out = int(self.cbed_size_scaled)
        b_even = int(nx_out % 2 != 0)
        nx = floor((nx_out-nx_in)/2)
        pads = [nx, nx+b_even, nx, nx+b_even]
        if nx_out/nx_in > 1:
            cbed_out = tf_pad2d(cbed_in, pads)
        else:
            cbed_out = cbed_in
        return cbed_out

    def find_center(self, nx):
        return np.ceil((nx + 1) / 2)

    @tf.function(jit_compile=True)
    def locate(self, x, y):
        xx = tf.math.round((x-1)*self.st_px*10.0)
        yy = tf.math.round((y-1)*self.st_px*10.0)
        x_int = tf.math.floor(xx/10.0)
        x_frac = tf.cast(xx % 10,tf.int8)
        y_int = tf.math.floor(yy/10.0)
        y_frac = tf.cast(yy % 10,tf.int8)
        return x_int, x_frac, y_int, y_frac

    def translate_idx(self):
        tt = np.zeros((self.le_y, self.le_x))
        tt[0:int(self.cbed_size_scaled),0:int(self.cbed_size_scaled)] = self.t
        return np.nonzero(tt == 1)

    @tf.function(jit_compile=True)
    def fcn_beam_shift_px(self, y, x):
        y = tf.cast(y,tf.complex64)
        x = tf.cast(x,tf.complex64)
        xlin = tf.math.exp(-self.rg*x)
        ylin = tf.math.exp(-self.rg*y)
        X,Y = tf.meshgrid(xlin, ylin)
        phase = X*Y
        center = phase.shape[0]//2
        return phase/phase[center, center]

    @tf.function(jit_compile=True)
    def scale_output(self, wave):
        beam_out_r = tf_complex_interpolation(wave,out_size=[self.cbed_size_scaled, self.cbed_size_scaled])
        return beam_out_r

    def update_obj(self, obj, coor):
        idx_bb = self.idx_bb[0]+int(coor[0]),self.idx_bb[1]+int(coor[1])
        if self.b_offset_correction:
            if np.all(coor == [0,0]):
                offset = 0
            else:
                phase_big = np.angle(self.object[idx_bb])
                phase = np.angle(obj)
                offset = np.sum((phase-phase_big))
                obj = np.exp(1j*(phase-offset*0.6))
        
        if self.__multithreading__:
            with self.__lock__:
                self.object[idx_bb] += obj
        else:
            self.object[idx_bb] += obj

    @tf.function(jit_compile=True)
    def shift_obj_patch(self, obj, y_frac, x_frac):
        phase_ramp = \
            self.phase_ramp_lib[0,y_frac] + \
            self.phase_ramp_lib[0,x_frac]
        # phase_ramp = self.fcn_beam_shift_px(y_frac, x_frac)
        obj = tf_ifft2d(obj * phase_ramp)
        obj = tf.gather_nd(obj,self.idx_b_sc_tf)
        return obj

    def update_patch(self, data, yx):

        yxb = tf.cast(yx,tf.float32)
        x_int, x_frac, y_int, y_frac = self.locate(yxb[:,1],yxb[:,0])
    
        for b in range(data.shape[0]):
            # obj_patch = self.shift_obj_patch(self.tf_pad_cbed(tf_fft2d(data[b,...]*self.weight)), y_frac[b], x_frac[b])
            obj_patch = self.tf_pad_cbed(self.shift_obj_patch(tf_fft2d(data[b,...]*self.weight), y_frac[b], x_frac[b]))
            self.update_obj(obj_patch.numpy(), [y_int[b], x_int[b]])

            if self.ds_ews is not None:
                self.ds_write_to_disk(data,b)
            with self.__lock__:
                self.y_pos = np.max([self.y_pos, int(y_int[b])])

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

        im = list()

        im.append(axes[0,0].imshow(np.abs(pred)**0.2))
        im.append(axes[0,1].imshow(np.angle(pred)))
        im.append(axes[1,0].imshow(np.abs(self.beam_r)))
        im.append(axes[1,1].imshow(np.angle(self.beam_r)))
        im.append(axes[2,0].imshow(np.abs(obj)))
        im.append(axes[2,1].imshow(np.angle(obj)))

        for ix, axi in enumerate(axes.ravel()):
            axi.set_axis_off()
            axi.set_title(cols[ix],fontdict=font)
                
        plt.tight_layout()
        plt.savefig('prediction_a.png')
        plt.close(fig)

def update_obj_fig(worker, obj_fig, fig):
    le_half = int(worker.cbed_size_scaled//2)
    data = np.angle(worker.object[le_half:-(le_half),le_half:-(le_half)])
    # data[worker.y_pos:,:] = np.core.nan
    obj_fig.set_data(data)
    obj_fig.set_clim(np.nanmin(data), np.nanmax(data))
    fig.canvas.flush_events() 

def plot_set_init(worker):
    le_half = int(worker.cbed_size_scaled//2)
    set_fig, set_ax = plt.subplots(3,5)
    set_ax_obj = []
    for iy in range(3):
        for ix in range(3):
            set_ax_obj.append(set_ax[iy, ix].imshow(np.ones((128,128))))
    
    set_ax_obj.append(set_ax[0, 3].imshow(np.ones((128,128))))
    set_ax_obj.append(set_ax[0, 4].imshow(np.ones((128,128))))
    set_ax_obj.append(set_ax[1, 3].imshow(np.ones((128,128))))
    set_ax_obj.append(set_ax[1, 4].imshow(np.ones((128,128))))

    set_ax_obj.append(set_ax[2, 3].imshow(np.angle(worker.object[le_half:-(le_half),le_half:-(le_half)]),vmin=-pi, vmax=pi))
    set_ax_obj.append(set_ax[2, 4].imshow(np.ones((128,128))))
    [axi.set_axis_off() for axi in set_ax.ravel()]
    set_fig.tight_layout()
    return set_fig, set_ax_obj

def plot_set_update(set_ax_obj, set_fig, set, pred, pos, worker, x_o):
        order = [8, 5, 2, 7, 4, 1, 6, 3, 0]
        # order = np.flip(order)
        le_half = int(worker.cbed_size_scaled//2)
        pos = tf.cast(pos,tf.float32)
        for ib, set_b in enumerate(set['cbeds']):
            if (ib+1) == len(set['cbeds']):
                for ix, ix_s in enumerate(order):
                    data = set_b[...,ix]**0.1
                    set_ax_obj[ix_s].set_data(data)
                    set_ax_obj[ix_s].set_clim(np.min(data), np.max(data))
            
        ew_r = (x_o[0,...])
        ew_k = tf_fft2d(x_o[0,...])

        set_ax_obj[9].set_data(np.angle(ew_r))
        set_ax_obj[9].set_clim(-pi, pi)
        ew_r_a = np.abs(ew_r)
        set_ax_obj[10].set_data(ew_r_a)
        set_ax_obj[10].set_clim(np.min(ew_r_a), np.max(ew_r_a))

        set_ax_obj[11].set_data(np.angle(ew_k))
        set_ax_obj[11].set_clim(-pi, pi)
        ew_k_a = np.abs(ew_k)
        set_ax_obj[12].set_data(ew_k_a)
        set_ax_obj[12].set_clim(np.min(ew_k_a), np.max(ew_k_a))

        obj_glob = np.angle(worker.object[le_half:-(le_half),le_half:-(le_half)])
        obj_glob[worker.y_pos:,:] = np.core.nan
        set_ax_obj[13].set_data(obj_glob)
        set_ax_obj[13].set_clim(np.nanmin(obj_glob), np.nanmax(obj_glob))
        
        msk = np.zeros_like(ew_r, dtype=float)
        msk[worker.idx_b] = 1.0
        set_ax_obj[14].set_data(np.angle(ew_r*worker.weight)*msk)
        set_ax_obj[14].set_clim(-np.pi, np.pi)
        set_fig.canvas.flush_events() 


def retrieve_phase_from_generator(ds_class, prms_net, options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None, 'batch_size':32}, model=None, live_update=True):
    if 'batch_size' not in options:
        options['batch_size'] = 32
    if 'threads' not in options:
        options['threads'] = 1
    if 'b_offset_correction' not in options:
        options['b_offset_correction'] = False
    if 'ew_ds_path' not in options:
        options['ew_ds_path'] = None
    
    steps= np.cast[np.int32](np.ceil(ds_class.ds_n_dat/options['batch_size']))
    worker = ReconstructionWorker(64, ds_class.rec_prms, options)
    le_half = int(worker.cbed_size_scaled//2)

    if live_update:
        plt.ion()
        fig, ax = plt.subplots()
        obj_fig = ax.imshow(np.angle(worker.object[le_half:-le_half,le_half:-le_half]),vmin=-pi, vmax=pi, cmap=parula)
        # set_fig, set_ax = plot_set_init(worker)

    if model is None:
        model = load_model(prms_net, ds_class.ds._flat_shapes[0], options['batch_size'],ds_class.rec_prms['beam_in_r'])

    ds_iter = iter(ds_class.ds.batch(options['batch_size'], drop_remainder=False).prefetch(8))
    
    t= tqdm(unit=' samples', total=ds_class.ds_n_dat)
    set = next(ds_iter)
    # set['cbeds'] = np.cast['uint8'](set['cbeds'])
    for _ in range(steps):
        set = next(ds_iter)
        pred = model.predict_on_batch(set['cbeds'])
        if worker.__multithreading__:
            worker.ThreadPool.submit(worker.update_patch, pred, set['pos'])
        else:
            worker.update_patch(pred, set['pos'])
        if live_update: 
            update_obj_fig(worker, obj_fig, fig)
            # plot_set_update(set_ax, set_fig, set, pred, set['pos'], worker, x_o)
        t.update(pred.shape[0])
       
    if worker.__multithreading__:
        while not worker.ThreadPool._work_queue.empty():
            if live_update:
                update_obj_fig(worker, obj_fig, fig)
            time.sleep(0.1)
        worker.ThreadPool.shutdown(True)
    t.close()
    if live_update:
        update_obj_fig(worker, obj_fig, fig)
        plt.colorbar(obj_fig,ax=ax)
        plt.show(block=True)

    return worker.object[le_half:-(le_half),le_half:-(le_half)]
    