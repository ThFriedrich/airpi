#%%
import h5py
import numpy as np
import pixstem.api as ps
import scipy.io as sio
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import matplotlib.pyplot as plt
import cv2 as cv

def fcn_fix_pixel(ds):
    s_hot_pixels = ds.find_hot_pixels(show_progressbar=False)
    s_dead_pixels = ds.find_dead_pixels(show_progressbar=False)
    ds = ds.correct_bad_pixels(s_hot_pixels + s_dead_pixels)
    ds.compute()
    return ds

def fcn_shift_to_center(ds):
    pacbed = ds.sum(axis = (0,1))
    # edges = canny(pacbed.data, sigma=3, low_threshold=10, high_threshold=90)
    # hough_radii = np.arange(5, 200, 1)
    # hough_res = hough_circle(edges, hough_radii)
    # accums, cx, cy, radius = hough_circle_peaks(hough_res, hough_radii,
    #                                            total_num_peaks=1)
    vis = np.cast[np.uint8](pacbed.data/np.max(pacbed.data)*255)
    circle = cv.HoughCircles(vis,cv.HOUGH_GRADIENT,1,50,
                            param1=90,param2=10,minRadius=5,maxRadius=100)
    radius = 0
    if len(circle) > 0:
        cx = circle[0][0][0]
        cy = circle[0][0][1]
        radius = circle[0][0][2]                                           
        ds = ds.shift_diffraction(cx - round(ds.data.shape[2]/2), cy - round(ds.data.shape[3]/2), interpolation_order=0)
    return ds, radius

def preprocess(ds, angle, target_cbed_size): # angle in the direction opposite to the one in SSB parameter.m file
    # ds = fcn_fix_pixel(ds) # takes huge amount of memory
    # ds = ds.rotate_diffraction(angle, show_progressbar=True)
    # ds = ds.rebin((ds.data.shape[0], ds.data.shape[1], target_cbed_size, target_cbed_size))
    ds, radius = fcn_shift_to_center(ds)
    # if len(radius) > 0: 
    #     adf = ds.virtual_annular_dark_field(round(ds.data.shape[2]/2), round(ds.data.shape[3]/2), radius+2, round(ds.data.shape[2]/2))
    #     adf.plot()
    #     return ds, adf
    # else:
    return ds, None
    # adf.plot()
    

#%% DEFINE READ/WIRTE DATA
# hf_read = h5py.File('/media/thomas/SSD/Samples/WS/ws2.h5', 'r')
# hf_read = h5py.File('/media/thomas/SSD/Samples/zeolite_frame/zeo_dose_3.h5', 'r')
# hf_read = h5py.File('/media/thomas/SSD/Samples/STO/hole3.h5', 'r')
hf_read = h5py.File('/media/thomas/SSD/Samples/Au/Au_NP.hdf5', 'r')
ds_read = hf_read.get('data')
rx, ry, kx, ky = ds_read.shape
rx = 1000
x_chunk = rx//100
hf_write = h5py.File('/media/thomas/SSD/Samples/Au/Au_NP.h5', 'w')
ds_write = hf_write.create_dataset('ds', (rx,ry,128,128), chunks = (x_chunk,ry,128,128),dtype=ds_read.dtype)

for x in range(0,rx,x_chunk):
    ds = np.array(ds_read[x:x+x_chunk,:])
    ds = ps.PixelatedSTEM(ds)
    ds, adf = preprocess(ds,0,128)
    ds.data[ds.data<0] = 0
    ds_write[x:x+x_chunk,:] = ds.data

# adict = {}
# adict['ds'] = np.transpose(ds.data, (3,2,1,0))
# sio.savemat('p_sto.mat', adict)

        
        
        
    