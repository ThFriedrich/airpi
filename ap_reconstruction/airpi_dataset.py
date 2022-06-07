from matplotlib import pyplot as plt
import numpy as np
import h5py
import scipy.io as sio
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import warnings

from tqdm import tqdm
from airpi.ap_reconstruction.tf_generator import tf_generator_ds, np_sequential_generator
from airpi.ap_architectures.utils import bin_mask, mrad_2_rAng
import tensorflow as tf
class airpi_dataset:
    def __init__(self, rec_prms, file, key='fpd_expt/fpd_data/data', dose=None, in_memory=False, step=1):
        # Raw Dataset properties
        self.path = file
        self.key = key
        self.in_memory_overwrite = in_memory
        # Parameters relevant for Reconstruction
        self.rec_prms = rec_prms

        # Dataset Augmentation
        self.dose = dose
        self.step = step

        # Initialization routines
        self.load_ds()
        self.get_bfm(rec_prms['bfm_type'])
        
            
    def get_ds_type(self):
        self.in_memory = True
        if self.path.endswith('.npy'):
            self.type = 'np'
        elif    self.path.endswith('.mat') or \
                self.path.endswith('.h5') or \
                self.path.endswith('.hdf5') or \
                self.path.endswith('.hd5'):
            if h5py.is_hdf5(self.path):
                self.type = 'h5'  
                if not self.in_memory_overwrite:
                    self.in_memory = False
            else:
                self.type = 'ma'
        else:
            self.type = None

    def get_hd_dataset_keys(self, f):
        keys = []
        f.visit(lambda key : keys.append(key) if type(f[key]) is h5py._hl.dataset.Dataset else None)
        return keys

    def load_ds(self):
        self.get_ds_type()
        scaling = None
        if self.type == 'h5':
            '''Gets hdf5-File handle and creates a generator object as tensorflow input'''
            hd_ds = h5py.File(self.path, 'r')
            ds_keys = self.get_hd_dataset_keys(hd_ds)
            key_id = ds_keys[ds_keys.index(self.key)]
            data = hd_ds[key_id]
        elif self.type == 'ma':  
            '''Loads matlab data into memory and creates a generator object as tensorflow input'''  
            matfile = sio.loadmat(self.path)
            data = matfile[self.key].astype(np.float32)
        elif self.type == 'np':
            '''Loads numpy data into memory and creates a generator object as tensorflow input'''
            data = np.load(self.path).astype(np.float32)
        else:
            warnings.warn("Warning! No valid Datset format detected!")

        self.rec_prms['nx']  = np.floor(data.shape[1]/self.step).astype(np.int32)
        self.rec_prms['ny']  = np.floor(data.shape[0]/self.step).astype(np.int32)
        self.ds_n_dat = self.rec_prms['ny'] * self.rec_prms['nx']
        self.ds_dims = (self.rec_prms['ny'] , self.rec_prms['nx'], data.shape[2], data.shape[3])
        self.ds_seq = np_sequential_generator(data, self.ds_dims, self.step*5, scaling)
        self.get_bfm(self.rec_prms['bfm_type']) 
        self.ds = tf_generator_ds(data, self.ds_dims, self.in_memory, self.rec_prms['beam_in_k'], self.step, self.dose)
        
    
    def get_bfm(self, bfm_type, beam_in = None): 

        if bfm_type == 'gene':
            self.rec_prms['beam_in_k'],  self.rec_prms['beam_in_r']  = np.array(bin_mask(self.rec_prms['E0'] , self.rec_prms['apeture'] , self.rec_prms['gmax'] , self.ds_dims[2], self.rec_prms['aberrations'] , 'b', 'complex'))
            # self.rec_prms['beam_in_k'] , self.rec_prms['beam_in_r']  = np.array(bin_mask(self.rec_prms['E0'] , self.rec_prms['apeture'] , self.rec_prms['gmax'] , 128, [-1, 0.001], 'b', 'complex'))

        elif bfm_type == 'avrg':
            pacbed = np.zeros(self.ds_dims[2:])
            for cbed in tqdm(self.ds_seq):
                pacbed += cbed
            pacbed /= np.amax(pacbed)
            edges = canny(pacbed, sigma=1)
            hough_radii = np.arange(5, 200, 1)
            hough_res = hough_circle(edges, hough_radii)
            accums, cx, cy, radius = hough_circle_peaks(hough_res, hough_radii,
                                                    total_num_peaks=1)
            self.rec_prms['gmax'] = np.cast[np.float32](mrad_2_rAng(self.rec_prms['E0'],self.rec_prms['apeture'])*(self.ds_dims[2]/2/radius[0]))
            msk = pacbed < 0.1
            beam_in_f = np.cast[np.complex](np.where(msk,0.0,1.0))
            est_beam  = np.array(bin_mask(self.rec_prms['E0'] , self.rec_prms['apeture'] , self.rec_prms['gmax'] , self.ds_dims[2], self.rec_prms['aberrations'] , 'k', 'complex'))
            # self.rec_prms['beam_in_k']  = np.stack([np.abs(beam_in_f), est_beam[...,1]],axis=-1)
            self.rec_prms['beam_in_k'] = est_beam
        else:
            self.bfm = None
        

    


    
