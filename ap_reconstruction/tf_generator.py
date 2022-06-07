import numpy as np
from threading import Thread
from multiprocessing import Queue, Process, Event
import matplotlib.pyplot as plt
from tensorflow.data import Dataset as tf_ds
from tensorflow.random import poisson as tf_poiss
from tensorflow import int32, float32, squeeze, math

class tf_generator:
    '''Generator that keeps the HDF5-DB open and reads chunk-by-chunk'''
    def __init__(self, data, dims, bfm, step, dose):
        self.file = data
        self.size_in =  dims
        self.dose = dose
        self.d_type = self.file.dtype 
        self.bfm = bfm
        self.step = step
        self.x_rest = -(self.file.shape[0] % self.step)
        if self.x_rest == 0:
            self.x_rest = None
        self.rw = np.zeros((self.size_in[0]+2,) + (3,) + self.size_in[2:],dtype=self.d_type)
        load_dat = self.file.astype(self.d_type)._dset[:self.x_rest:self.step,0,...]
        if self.dose is not None:
            load_dat = squeeze(tf_poiss([1], load_dat*self.dose, seed=131087))
        self.rw[1:self.size_in[0]+1,1,...] = load_dat
        self.rw[1:self.size_in[0]+1,2,...] = load_dat
        
        self.q_row = Queue(maxsize=8)
        self.q_set = Queue(maxsize=256)

        self.rows_finished = Event()
        self.finished = Event()

        self.rw_thread = Thread(target=self.RowWorker,daemon=True, name='RowWorker')
        self.co_thread = Thread(target=self.ColumnWorker,daemon=True, name='ColumnWorker')
        self.rw_thread.start()
        self.co_thread.start()
        
    def RowWorker(self): 
        for rw_i in range(1,self.size_in[1]):
            self.rw = np.roll(self.rw,-1,axis=1)
            load_dat = self.file.astype(self.d_type)._dset[:self.x_rest:self.step,(rw_i-1)*self.step,...] 
            if self.dose is not None:
                load_dat = squeeze(tf_poiss([1], load_dat*self.dose))
            self.rw[1:self.size_in[0]+1,2,...] = load_dat
            self.q_row.put((self.rw, rw_i))

        # Pad last row with empty cbeds
        self.rw = np.roll(self.rw,-1,axis=1)
        self.rw[1:self.size_in[0]+1,2,...] = load_dat
        # self.rw[:,2,...] = 0
        self.q_row.put((self.rw, rw_i+1))
        self.rows_finished.set()


    def ColumnWorker(self):  
        while True:
            rw, ri = self.q_row.get()
            rw[0,...] = rw[1,...]
            rw[-1,...] = rw[-2,...]
            for rx in range(1, self.size_in[0]+1):
                self.q_set.put((self.cast_set(rw, rx) , rx, ri))
            if self.rows_finished.is_set(): 
                if self.q_row.qsize()==0:
                    self.finished.set()
                    break
            
    def cast_set(self, rw, rx):
        set = np.cast[np.float32](rw[rx-1:rx+2,...])
        set = np.transpose(np.squeeze(set),(2,3,0,1))
        set = np.flip(set,3)
        set = np.reshape(set, self.size_in[2:] + (9,))**2
        if self.bfm.all() is not None:
            set = np.concatenate((set,self.bfm),-1)
        return set
     

    def __call__(self):  
        while True:
            if self.finished.is_set(): 
                if self.q_set.qsize()==0:
                    break
            set, ix, iy = self.q_set.get()
            yield {'cbeds': set, 'pos': np.expand_dims([iy, ix],-1)}
                    
class tf_generator_in_memory:
    def __init__(self, data, size_in, bfm, step, dose):
        self.bfm = bfm
        self.d_type = data.dtype 
        self.data = data.astype(self.d_type)._dset[...]
        self.data = np.pad(self.data,((1,1),(1,1),(0,0),(0,0)),mode='reflect')
        self.step = step
        self.dose = dose
        self.rn_x = list(range(1,size_in[0]+1,self.step))
        self.rn_y = list(range(1,size_in[1]+1,self.step))
       
    def cast_set(self, rw, rx):
        rx_st = rx-self.step; rxpst = rx+self.step
        ry_st = rw-self.step; rypst = rw+self.step
        load_list = np.array(
            [[rx_st,ry_st], [rx_st,rw],  [rx_st,rypst], 
            [rx,ry_st],     [rx,rw],     [rx,rypst], 
            [rxpst,ry_st],  [rxpst,rw],  [rxpst,rypst]],dtype=np.int32)
        set = self.data[load_list[:,0],load_list[:,1]].transpose([1,2,0])
        
        if self.dose is not None:
            set = squeeze(tf_poiss([1], set*self.dose))
        if self.bfm is not None:
            set = np.concatenate((set,self.bfm),-1)

        return set
    
    def __plot_set__(self, set):
        fig = plt.figure()
        order = [7, 4, 1, 8, 5, 2, 9, 6, 3]
        for ix, ix_s in enumerate(order):
            ax = fig.add_subplot(3, 3, ix_s)
            ax.imshow(set[...,ix]**0.1)
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig('set.png')
        plt.close()

    def __call__(self):  
        for iy in self.rn_y:
            for ix in self.rn_x:
                set = self.cast_set(iy,ix)
                # self.__plot_set__(set)
                yield {'cbeds': set, 'pos': np.expand_dims([iy, ix],-1)}


def tf_generator_ds(data, size_in, b_memory, bfm=None, step=1, dose=None):
    if bfm is not None:
        nc = 9 + bfm.shape[-1]
    else:
        nc = 9
    if b_memory:
            return tf_ds.from_generator(
            tf_generator_in_memory(data, size_in, bfm, step, dose),
            output_types=(  {'cbeds':float32, 'pos': int32}),
            output_shapes=( {'cbeds': [size_in[2], size_in[3], nc], 'pos':[2,1]} ) 
            )
    else:
        return tf_ds.from_generator(
            tf_generator(data, size_in, bfm, step, dose),
            output_types=(  {'cbeds':float32, 'pos': int32}),
            output_shapes=( {'cbeds': [size_in[2], size_in[3], nc], 'pos':[2,1]} ) 
            )

class np_sequential_generator():
    '''Generator that keeps the HDF5-DB open and reads chunk-by-chunk'''
    def __init__(self, data, dims, step, scaling=None):
        self.file = data
        self.size_in = dims
        self.step = step
        self.d_type = self.file.dtype 
        self.rw = self.file.astype(self.d_type)._dset[:,1,...]        
        self.scale_ds = scaling
        if self.scale_ds is not None:
            sc = self.scale_ds.astype(self.d_type)._dset[:,y,...]
            rw = (np.cast[np.float](self.rw)*sc)/65536**10
    def __iter__(self):
        for y in range(1,self.size_in[1],self.step):
            rw = self.file.astype(self.d_type)._dset[:,y,...]
            if self.scale_ds is not None:
                sc = self.scale_ds.astype(self.d_type)._dset[:,y,...]
                rw = (np.cast[np.float](rw)*sc)/65536**10
            for x in range(1,self.size_in[0],self.step):
                yield np.cast[np.float32](rw[x,...])
   
            
# class generator_naive:
#     '''Generator that keeps the HDF5-DB open and reads chunk-by-chunk'''
#     def __init__(self, file, key):
#         self.file = h5py.File(file, 'r')[key]
#         self.size_in =  self.file.shape
#         self.d_type = self.file.dtype 

#     def cast_set(self, set):
#         out = tf.cast(set,tf.float32)
#         out = tf.transpose(tf.squeeze(out),(2,3,0,1))
#         out = tf.reverse(out,[3])
#         out = tf.reshape(out, self.size_in[2:] + (9,))
#         return out
    
#     def __plot_set__(self, set):
#         fig = plt.figure()
#         order = [7, 4, 1, 8, 5, 2, 9, 6, 3]
#         for ix, ix_s in enumerate(order):
#             ax = fig.add_subplot(3, 3, ix_s)
#             ax.imshow(set[...,ix].numpy()**0.1)
#             ax.set_axis_off()
#         plt.tight_layout()
#         plt.savefig('set.png')

#     def __call__(self):  
#         for cl_i in range(1,self.size_in[0]-1):
#             for rw_i in range(1,self.size_in[1]-1):
#                 set = self.cast_set(self.file.astype(self.d_type)._dset[(cl_i-1):(cl_i+2),(rw_i-1):(rw_i+2),...])
#                 yield {'cbeds': set, 'pos': np.expand_dims([rw_i, cl_i],-1)}
