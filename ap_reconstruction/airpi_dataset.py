import numpy as np
import h5py
import scipy.io as sio
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import warnings
# import dask.array as da

from ap_reconstruction.tf_generator import tf_generator_ds, np_sequential_generator
from ap_utils.functions import tf_normalise_to_one_complex, tf_probe_function, tf_mrad_2_rAng
import tensorflow as tf

class airpi_dataset:
    def __init__(self, rec_prms, file, key='/ds', dose=None, skip=1):
        # Raw Dataset properties
        self.path = file
        self.key = key

        # Parameters relevant for Reconstruction
        self.rec_prms = rec_prms

        # Dataset Augmentation
        self.dose = dose
        self.skip = skip

        if 'order' not in self.rec_prms:
            self.rec_prms['order'] = ['rx','ry','kx','ky']
        if 'skip' not in self.rec_prms:
            self.rec_prms['skip'] = 1
        if 'dose' not in self.rec_prms:
            self.rec_prms['dose'] = None
        if 'b_add_probe' not in self.rec_prms:
            self.rec_prms['b_add_probe'] = False
        if 'b_resize' not in self.rec_prms:
            if self.rec_prms['cbed_size'] != 64:
                self.rec_prms['b_resize'] = True
            else:
                self.rec_prms['b_resize'] = False
        if self.rec_prms['b_resize']:
            self.rec_prms['cbed_size'] = 64
        # Initialization routines
        self.load_ds()

    def get_ds_type(self):
        if self.path.endswith('.npy'):
            self.type = 'np'
        elif    self.path.endswith('.mat') or \
                self.path.endswith('.h5') or \
                self.path.endswith('.hdf5') or \
                self.path.endswith('.hd5'):
            if h5py.is_hdf5(self.path):
                self.type = 'h5'  
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
            if 'probe' in ds_keys:
                self.pacbed = hd_ds['probe'].astype(np.float32)._dset
            else:
                self.pacbed = None
        else:
            warnings.warn("Warning! No valid Datset format detected!")

        n_d1  = np.floor(data.shape[0]/self.skip).astype(np.int32)
        n_d2  = np.floor(data.shape[1]/self.skip).astype(np.int32)
        self.ds_n_dat = n_d1 * n_d2
        self.ds_dims = (n_d1 , n_d2, data.shape[2], data.shape[3])
        bfm_skip = np.max((1,self.ds_n_dat//100000))
        self.ds_seq = np_sequential_generator(data, self.ds_dims, bfm_skip, scaling, order=self.rec_prms['order'])
        self.get_probe(self.rec_prms['probe_estimation_method']) 
        # data = da.from_array(data, chunks=(3, 3, data.shape[2], data.shape[3]))
        if self.rec_prms['order'][0:2] == ['ry','rx']:
            self.rec_prms['nx'] = n_d2
            self.rec_prms['ny'] = n_d1
        elif self.rec_prms['order'][0:2] == ['rx','ry']:
            self.rec_prms['nx'] = n_d1
            self.rec_prms['ny'] = n_d2
        
        self.ds = tf_generator_ds(data, self.ds_dims, self.rec_prms)


    
    def get_probe(self, bfm_type): 

        if bfm_type == 'gene':
            pk = tf_probe_function(self.rec_prms['E0'] , self.rec_prms['apeture'] , self.rec_prms['gmax'] , self.rec_prms['cbed_size'], self.rec_prms['aberrations'], domain='r', type='complex',refine=True)
            pk = tf_normalise_to_one_complex(pk)
            self.rec_prms['probe'] = np.array(tf.stack([tf.math.abs(pk), tf.math.angle(pk)], axis=-1))
        elif bfm_type == 'avrg':
            if self.pacbed is None:
                self.pacbed = np.zeros(self.ds_dims[2:])
                for cbed in self.ds_seq:
                    self.pacbed += cbed
            self.pacbed /= np.amax(self.pacbed)
            edges = canny(self.pacbed, sigma=2)
            hough_radii = np.arange(5, 200, 1)
            hough_res = hough_circle(edges, hough_radii)
            _, _, _, radius = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
            self.rec_prms['gmax'] = np.cast[np.float32](tf_mrad_2_rAng(self.rec_prms['E0'],self.rec_prms['apeture'])*(self.ds_dims[2]/2/(radius[0]+1)))
            
            probe = tf_probe_function(self.rec_prms['E0'] , self.rec_prms['apeture'] , self.rec_prms['gmax'] , self.rec_prms['cbed_size'], self.rec_prms['aberrations'], domain='r', type='complex',refine=True)
            probe = tf_normalise_to_one_complex(probe)
            self.rec_prms['probe']  = np.array(tf.stack((tf.math.abs(probe), tf.math.angle(probe)),axis=-1))
        else:
            self.bfm = None
        

    


    
