import queue
import numpy as np
from threading import Thread
from multiprocessing import Event
import matplotlib.pyplot as plt
from ap_training.data_fcns import fcn_weight_cbeds
from tensorflow.data import Dataset as tf_ds
from tensorflow.random import poisson as tf_poiss
from tensorflow import int32, float32, squeeze, image, transpose, uint8

class tf_generator:
    '''Generator that keeps the HDF5-DB open and reads chunk-by-chunk'''
    def __init__(self, data, dims, bfm, step, dose, step_size, order, b_add_probe, b_resize):
        self.file = data
        self.size_in =  dims
        self.dose = dose
        self.d_type = self.file.dtype 
        self.bfm = bfm
        self.step = step
        self.step_size = step_size
        self.__set_row_getter__(order)
        self.add_probe = b_add_probe
        self.resize = b_resize
        self.n_threads = 4

        if self.x_rest == 0:
            self.x_rest = None
        self.rw = np.zeros((self.nx+2,) + (3,) + self.size_in[2:],dtype=self.d_type)
        load_dat = self.__get_row__(0)
        if self.dose is not None:
            load_dat = squeeze(tf_poiss([1], load_dat*self.dose, seed=131087))
        self.rw[1:self.nx+1,1,...] = load_dat
        self.rw[1:self.nx+1,2,...] = load_dat
        
        self.q_row = queue.Queue(maxsize=32)
        self.q_set = queue.Queue(maxsize=2048)

        self.rows_finished = Event()
        self.finished = Event()

        self.rw_thread = Thread(target=self.RowWorker,daemon=True, name='RowWorker')
        self.rw_thread.start()

        self.co_thread = []
        for i in range(self.n_threads):
            self.co_thread.append(Thread(target=self.ColumnWorker,daemon=True, name='ColumnWorker_'+str(i)))
            self.co_thread[i].start()

    def RowWorker(self): 
        for rw_i in range(1,self.ny):
            self.rw = np.roll(self.rw,-1,axis=1)
            load_dat = self.__get_row__(rw_i)
            if self.dose is not None:
                load_dat = squeeze(tf_poiss([1], load_dat*self.dose, seed=131087))
            self.rw[1:self.nx+1,2,...] = load_dat
            self.q_row.put((self.rw, rw_i))

        # Pad last row with copied cbeds
        self.rw = np.roll(self.rw,-1,axis=1)
        self.rw[1:self.nx+1,2,...] = load_dat
        self.q_row.put((self.rw, rw_i+1))
        self.rows_finished.set()

    def ColumnWorker(self):  
        while True:
            rw, ri = self.q_row.get()
            rw[0,...] = rw[1,...]
            rw[-1,...] = rw[-2,...]
            for rx in range(1, self.nx+1):
                self.q_set.put((self.cast_set(rw[rx-1:rx+2,...]) , rx, ri))
            self.q_row.task_done()
            if self.rows_finished.is_set(): 
                if self.q_row.unfinished_tasks==0:
                    self.finished.set()
                    break

    def __get_row_yx__(self,y):
                return self.file.astype(self.d_type)._dset[y*self.step,:self.x_rest:self.step,...]

    def __get_row_xy__(self,y):
                return self.file.astype(self.d_type)._dset[:self.x_rest:self.step, y*self.step,...]

    def __set_row_getter__(self, order):
        if order[0:2] == ['ry','rx']:
            self.nx = self.size_in[1]
            self.ny = self.size_in[0]
            self.x_rest = -(self.file.shape[1] % self.step)
            self.__get_row__ = self.__get_row_yx__    
        elif order[0:2] == ['rx','ry']:
            self.nx = self.size_in[0]
            self.ny = self.size_in[1]
            self.x_rest = -(self.file.shape[0] % self.step)
            self.__get_row__ = self.__get_row_xy__ 

    def cast_set(self, set):
        set = np.transpose(np.squeeze(set),(3,2,0,1))
        set = np.flip(set,[0, 1])
        set = np.reshape(set, self.size_in[2:] + (9,))
        
        # set = np.cast[np.float32](set)
        # set = fcn_weight_cbeds(set, self.step_size)

        if self.resize:
            set = transpose(set,[2,0,1])
            set = squeeze(image.resize(set[...,np.newaxis], (64,64)))
            set = transpose(set,[1, 2, 0])

        if self.add_probe:
            set = np.concatenate((set,self.bfm),-1)

        return set 

    def __call__(self):  
        while True:
            if self.finished.is_set(): 
                if self.q_set.unfinished_tasks==0:
                    break
            set, ix, iy = self.q_set.get()
            self.q_set.task_done()
            yield {'cbeds': set, 'pos': np.expand_dims([iy, ix],-1)}


def tf_generator_ds(data, size_in, prm):
    if prm['b_add_probe']:
        nc = 9 + prm['probe'].shape[-1]
    else:
        nc = 9
    size_out = np.array(size_in)
    if prm['b_resize']:
        size_out[2:4] = 64
     
    return tf_ds.from_generator(
        tf_generator(data, size_in, prm['probe'], prm['skip'], prm['dose'], prm['step_size'], prm['order'], prm['b_add_probe'], prm['b_resize']),
        output_types=(  {'cbeds':data.dtype, 'pos': int32}),
        output_shapes=( {'cbeds': [size_out[2], size_out[3], nc], 'pos':[2,1]} ) 
        )

class np_sequential_generator():
    '''Generator that keeps the HDF5-DB open and reads chunk-by-chunk'''
    def __init__(self, data, dims, step, scaling=None, order=['rx','ry','kx','ky']):
        self.file = data
        self.size_in = dims
        self.step = step
        self.d_type = self.file.dtype 
        self.rw = self.file.astype(self.d_type)._dset[:,1,...]        
        self.scale_ds = scaling
        self.__set_row_getter__(order)
    
    def __get_row_yx__(self,y):
                return self.file.astype(self.d_type)._dset[y,...]

    def __get_row_xy__(self,y):
                return self.file.astype(self.d_type)._dset[:,y,...]

    def __set_row_getter__(self, order):
        if order[0:2] == ['ry','rx']:
            self.nx = self.size_in[1]
            self.ny = self.size_in[0]
            self.__get_row__ = self.__get_row_yx__    
        elif order[0:2] == ['rx','ry']:
            self.nx = self.size_in[0]
            self.ny = self.size_in[1]
            self.__get_row__ = self.__get_row_xy__ 

    def __iter__(self):
        for y in range(1,self.ny,self.step):
            rw = self.__get_row__(y)
            for x in range(1,self.nx,self.step):
                yield np.cast[np.float32](rw[x,...])