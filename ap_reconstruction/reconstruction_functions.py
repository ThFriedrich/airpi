import time
import numpy as np
from math import ceil, floor, pi
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
from tqdm import tqdm
from threading import Lock
from ap_utils.colormaps import parula
from ap_utils.functions import tf_cast_complex, tf_fft2d, tf_ifft2d, tf_normalise_to_one_complex, tf_pad2d, tf_probe_function
from ap_utils.BoundedThreadpool import BoundedExecutor

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
        model = ARCH(prms_net, X_SHAPE, deploy=True, bs=bs, probe=probe).build(bs)
        model.load_weights(prms_net['cp_path']+'/cp-best.ckpt')
        model.summary()
        return model
    else:
        warnings.warn("Warning! No Checkpoint found! The loaded model is untrained!")
     

class ReconstructionWorker():
    """Reconstruction worker with multithreaded, jit-compiled tensorflow processing pipeline"""
    def __init__(self, rec_prms, options={'threads':1,'batch_size':128}):
        self.rec_prms = rec_prms
        self.y_pos = 0
        self.counter = 0
        self.create_obj(self.rec_prms['cbed_size'])  
        self.phase_ramp_lib_y, self.phase_ramp_lib_x = self.create_phase_ramp_lib()
        self.__lock__ = Lock()
        self.ThreadPool = BoundedExecutor(16, options['threads'])


    def create_obj(self, cbed_size): # I think this function need some serious documentation
        self.rec_prms["space_size"] = cbed_size/(2*self.rec_prms["gmax"])
        self.rec_prms["scale"] = self.rec_prms["oversample"]*self.rec_prms["space_size"]/cbed_size/self.rec_prms["step_size"]

        if self.rec_prms["scale"] > 1:
            self.cbed_size_scaled = int(ceil(cbed_size * self.rec_prms["scale"]))
        else:
            self.cbed_size_scaled = int(cbed_size)

        self.rec_prms["px_size"] = self.rec_prms["space_size"]/self.cbed_size_scaled
        gmax_s = tf.cast(self.cbed_size_scaled/(2*self.rec_prms["space_size"]),tf.float32)

        self.rg = tf.cast(np.arange(int(cbed_size))*2.0*np.pi*1j/self.cbed_size_scaled, tf.complex64)
        self.st_px = tf.cast(self.rec_prms['step_size']/self.rec_prms['px_size'],tf.float32)
        self.cbed_center = int(np.ceil((self.cbed_size_scaled+1)/2))-1

        self.le_y = int(round(self.rec_prms['ny']*self.st_px.numpy())+self.cbed_size_scaled-1)
        self.le_x = int(round(self.rec_prms['nx']*self.st_px.numpy())+self.cbed_size_scaled-1)
        self.object = np.zeros((self.le_y, self.le_x), dtype='complex64')
        self.le_half = int(self.cbed_size_scaled//2)

        
        self.beam_r = tf_cast_complex(self.rec_prms['probe'][...,0], self.rec_prms['probe'][...,1])
        self.beam_r = tf_normalise_to_one_complex(self.beam_r)
        beam_r_int = np.abs(self.beam_r)**2

        self.beam_r_scaled = tf_probe_function(self.rec_prms['E0'], self.rec_prms['apeture'], gmax_s,
                                          self.cbed_size_scaled, self.rec_prms['aberrations'], domain='r', type='complex', refine=True)
        beam_r_sc_int = np.abs(self.beam_r_scaled)**2
        
        # where the beam has significant intensity
        self.idx_b_sc = np.nonzero(beam_r_sc_int>0.1*(np.max(beam_r_sc_int))) 
        self.idx_b_sc_tf = tf.transpose(tf.stack((self.idx_b_sc[0],self.idx_b_sc[1])))
        weight = beam_r_int
        self.weight = tf.cast(weight / np.sum(weight),tf.complex64)
        
        self.t = np.zeros((int(self.cbed_size_scaled), int(self.cbed_size_scaled)))
        self.t[self.idx_b_sc] = 1
        self.idx_bb = self.translate_idx()

    @tf.function(jit_compile=True)
    def tf_pad_obj(self, obj_in):
        nx_in = int(obj_in.shape[1])
        nx_out = int(self.cbed_size_scaled)
        b_even = int(nx_out % 2 != 0)
        nx = floor((nx_out-nx_in)/2)
        pads = [nx, nx+b_even, nx, nx+b_even]
        if nx_out/nx_in > 1:
            cbed_out = tf_pad2d(obj_in, pads)
        else:
            cbed_out = obj_in
        return cbed_out

    @tf.function(jit_compile=True)
    def locate(self, yx):
        yx = tf.cast(yx,tf.float32)
        xx = tf.math.round((yx[:,1]-1)*self.st_px*10.0)
        yy = tf.math.round((yx[:,0]-1)*self.st_px*10.0)
        x_int = tf.math.floor(xx/10.0)
        x_frac = tf.cast(xx % 10,tf.int32)
        y_int = tf.math.floor(yy/10.0)
        y_frac = tf.cast(yy % 10,tf.int32)
        return x_int, x_frac, y_int, y_frac

    def translate_idx(self):
        tt = np.zeros((self.le_y, self.le_x))
        tt[0:int(self.cbed_size_scaled),0:int(self.cbed_size_scaled)] = self.t
        return np.nonzero(tt == 1)

    def create_phase_ramp_lib(self):
        phase_ramp_lib_y = tf.TensorArray(tf.complex64,size=10)
        phase_ramp_lib_x = tf.TensorArray(tf.complex64,size=10)
        for shift in range(10):
            phase_ramp_lib_y = phase_ramp_lib_y.write(shift,
                self.fcn_beam_shift_px(shift, 0)) # shift y
            phase_ramp_lib_x = phase_ramp_lib_x.write(shift,
                self.fcn_beam_shift_px(0, shift)) # shift x
        return phase_ramp_lib_y.stack(), phase_ramp_lib_x.stack()

    def fcn_beam_shift_px(self, y, x):
        y = tf.cast(y*0.1,tf.complex64)
        x = tf.cast(x*0.1,tf.complex64)
        xlin = tf.math.exp(-self.rg*x)
        ylin = tf.math.exp(-self.rg*y)
        X,Y = tf.meshgrid(xlin, ylin)
        phase = X*Y
        center = phase.shape[0]//2
        return phase/phase[center, center]
    
    @tf.function(jit_compile=True)
    def weight_shift_patch_tensor(self, patch_tensor, x_frac, y_frac):
        phase_ramp_stack_y = tf.squeeze(tf.gather_nd(self.phase_ramp_lib_y, indices=[y_frac]))
        phase_ramp_stack_x = tf.squeeze(tf.gather_nd(self.phase_ramp_lib_x, indices=[x_frac]))
        patch_tensor = self.tf_pad_obj(tf_fft2d(patch_tensor*self.weight) * phase_ramp_stack_y * phase_ramp_stack_x)
        return tf_ifft2d(patch_tensor)

    @tf.function(jit_compile=True)
    def update_obj(self, obj, coor):
        idx_y = self.idx_bb[0]+coor[0]
        idx_x = self.idx_bb[1]+coor[1]
        indices = tf.cast(tf.stack([idx_y, idx_x], axis=2),tf.int32)
        temp_obj = tf.zeros(tf.shape(self.object), tf.complex64)
        temp_obj = tf.tensor_scatter_nd_add(temp_obj,indices,obj)
        return temp_obj

    @tf.function(jit_compile=True)
    def get_patches_and_indices(self,data,yx):
        x_int, x_frac, y_int, y_frac = self.locate(yx)
        patches = self.weight_shift_patch_tensor(data, x_frac, y_frac)
        # There's probably more elegant way to do this
        pathcesT = tf.transpose(patches,perm=[1,2,0])
        patches_sub = tf.transpose(tf.gather_nd(pathcesT,self.idx_b_sc_tf))
        idx_y = self.idx_bb[0]+y_int
        idx_x = self.idx_bb[1]+x_int
        indices = tf.cast(tf.stack([idx_y, idx_x], axis=2),tf.int32)
        return patches_sub, indices, y_int
    
    def update_patch(self, data, yx):
        bs = tf.shape(data)[0]
        patches_sub, indices, y_int = self.get_patches_and_indices(data, yx)
        with self.__lock__:
            self.y_pos = int(np.max([self.y_pos, np.max(y_int)]))
            self.counter += int(bs)
            self.object = tf.tensor_scatter_nd_add(self.object,indices,patches_sub)
        
def update_obj_fig(worker, obj_fig, fig):
    le_half = int(worker.cbed_size_scaled//2)
    data = np.angle(worker.object[le_half:-(le_half),le_half:-(le_half)])
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
        le_half = int(worker.cbed_size_scaled//2)
        pos = tf.cast(pos,tf.float32)

        set_b = set['cbeds'][0]
        for ix, ix_s in enumerate(order):
            data = np.cast[np.float32](set_b[...,ix])**0.1
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


def retrieve_phase_from_generator(ds_class, prms_net=None, options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None, 'batch_size':32}, model=None, live_update=True):
    if all(item is None for item in [prms_net, model]):
        raise ValueError("Either model parameters 'prms_net' or a loaded model 'model' must be provided. Both are received None")

    if 'batch_size' not in options:
        options['batch_size'] = 256
    if 'threads' not in options:
        options['threads'] = 1
    
    steps= np.cast[np.int32](np.ceil(ds_class.ds_n_dat/options['batch_size']))
    worker = ReconstructionWorker(ds_class.rec_prms, options)
    le_half = int(worker.cbed_size_scaled//2)

    if live_update:
        plt.ion()
        fig, ax = plt.subplots()
        obj_fig = ax.imshow(np.angle(worker.object[le_half:-le_half,le_half:-le_half]),vmin=-pi, vmax=pi, cmap=parula)
        # set_fig, set_ax = plot_set_init(worker)

    if model is None:
        model = load_model(prms_net, ds_class.ds._flat_shapes[0], options['batch_size'], ds_class.rec_prms['probe'])

    ds_iter = iter(ds_class.ds.batch(options['batch_size'], drop_remainder=False).prefetch(8))
    prev_count = 0
    t2= tqdm(unit=' samples', total=ds_class.ds_n_dat, desc="REC")
    t1= tqdm(unit=' samples', total=ds_class.ds_n_dat, desc="CNN")
    upd_tic = time.time()
    for _ in range(steps):
        set = next(ds_iter)
        pred = model.predict_on_batch(set['cbeds'])
        worker.ThreadPool.submit(worker.update_patch, pred, set['pos'])
        
        if live_update: 
            tic = time.time()
            if tic - upd_tic > 1:
                update_obj_fig(worker, obj_fig, fig)
                upd_tic = tic
                # plot_set_update(set_ax, set_fig, set, pred, set['pos'], worker, x_o)
        t1.update(pred.shape[0])
        t2.update(worker.counter-prev_count)
        prev_count = worker.counter
    t1.close()

    while not worker.ThreadPool.executor._work_queue.empty():
        if live_update:
            update_obj_fig(worker, obj_fig, fig)
        t2.update(worker.counter-prev_count)
        prev_count = worker.counter
        time.sleep(0.1)

    worker.ThreadPool.shutdown(True)
    time.sleep(0.05)
    t2.update(worker.counter-prev_count)
    t2.close()
    if live_update:
        update_obj_fig(worker, obj_fig, fig)
        plt.colorbar(obj_fig,ax=ax)
        plt.show(block=True)

    return worker.object[le_half:-(le_half),le_half:-(le_half)]
    
