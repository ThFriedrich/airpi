import os
import argparse
import silence_tensorflow.auto
import scipy.io as sio
import numpy as np
import time

def reconstruct_ds(ds_path, key, prms_net, rec_prms, options, skip=1, dose=None):
    rec_prms['step_size'] *= skip
    ds_class = airpi_dataset(rec_prms, ds_path, key, dose, skip=skip)
    return retrieve_phase_from_generator(ds_class, prms_net, options, live_update=True)

def get_model_ckp(cp_path):
    hp_file = os.path.join(cp_path, "hyperparameters.pickle")
    _, prms_net = load_hparams(hp_file)
    prms_net["cp_path"] = cp_path
    return prms_net

if __name__ == "__main__":
    os.system("clear")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", type=int, default=0, help="Dose")
    parser.add_argument("--ds", type=int, default=0, help="Dataset")
    parser.add_argument("--skip", type=int, default=1, help="Step skip")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU")
    # parser.add_argument("--model", type=str, default='UNET_24_D3_sk', help="GPU")
    parser.add_argument("--model", type=str, default='V_32_D3_sk_r10', help="GPU")
    parser.add_argument("--ap_fcn", type=str, default='avrg', help="Aperture function estimation, gene: parameter generated, avrg: use PACBED")
    
    args = vars(parser.parse_args())
    dose = int(args["dose"])
    ds = int(args["ds"])
    if dose == 0:
        dose = None

    cp_path = 'Ckp/Training/' + args["model"]
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])

    from tensorflow import config as tf_config, debugging
    tf_config.optimizer.set_jit("autoclustering")
    from ap_reconstruction.reconstruction_functions import (
        retrieve_phase_from_generator,
    )
    debugging.disable_traceback_filtering()
    # debugging.disable_check_numerics()

    from ap_reconstruction.airpi_dataset import airpi_dataset
    from ap_utils.file_ops import load_hparams
    from ap_utils.globals import debugger_is_active
    tf_config.run_functions_eagerly(debugger_is_active())
    prms_net = get_model_ckp(cp_path)

    # twisted bilayer_graphene
    if ds == 0:
        hd5_in = "/media/thomas/SSD/Samples/graphene/gra.hdf5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 200.0,
            "apeture": 25,
            "gmax": 2.5,
            "cbed_size": 128,
            "step_size": 0.2,
            "aberrations": [-1, 0.001],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'threads':1, 'batch_size':256}
    
    if ds == 1:
        hd5_in = "/media/thomas/SSD/Samples/graphene/gra_tim.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 60.0,
            "apeture": 34,
            "gmax": 1.8632454,
            "cbed_size": 64,
            "step_size": 0.04,
            "aberrations": [0, 0],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1.0,
            "order": ['ry','rx','kx','ky']
        }
        options={'threads':2, 'batch_size':256}

    if ds == 2:
        hd5_in = "/media/thomas/SSD/Samples/STO/sto_sro.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20,
            "gmax": 1.6253973,
            "cbed_size": 64,
            "step_size": 0.43,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2.0,
            "order": ['ry','rx','kx','ky']
        }
        options={'threads':2, 'ew_ds_path':None}
    
    if ds == 3:
        hd5_in = "/media/thomas/SSD/Samples/MgO/MgO.h5"
        hd5_key = "ds_cbed"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20,
            "gmax": 1.6253973,
            "cbed_size": 64,
            "step_size": 0.05,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample":2.0,
            "order": ['rx','ry','kx','ky']
        }
        options={'threads':2, 'batch_size':128}
    
    if ds == 4:
        # hd5_in = "/media/thomas/SSD/Samples/Zeolite/64x64x1000x1000_processed_cheat.hdf5"
        # hd5_in = "/media/thomas/SSD/Samples/Zeolite/64x64x1000x1000_processed.hdf5"
        hd5_in = "/media/thomas/SSD/Samples/Zeolite/64x64x499x499_aligned_uin8_cheat5.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 200.0,
            "apeture": 12,
            "gmax": 0.76557,
            "cbed_size": 64,
            "step_size": 0.3,
            "aberrations": [-1, 1e-3],
            # "aberrations": [0, 0],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1.0,
            "order": ['rx','ry','kx','ky']
        }
        options={'threads':8, 'batch_size': 128}

    # MoS2ew_ds_path
    if ds == 5:
        hd5_in = "/media/thomas/SSD/Samples/MSO/airpi_sto.h5"
        hd5_key = "ds_int"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 4.5714,
            "cbed_size": 128,
            "step_size": 0.05,
            # "aberrations": [-1, 1e-3],
            "aberrations": [14.0312, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1.0,
        }
        options={'threads':1, 'ew_ds_path':None}

    if ds == 6:
        # STO
        hd5_in = "/media/thomas/SSD/Samples/STO/hole_preprocessed_cropped_2.h5"
        # hd5_in = "/media/thomas/SSD/Samples/STO/hole.h5"
        # hd5_in = "/media/thomas/SSD/Samples/STO/hole3_64x64_bin_cheat.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 1.6671,
            "cbed_size": 64,
            "step_size": 0.1818*2,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1.0,
            "order": ['rx','ry','kx','ky']
        }
        options={'threads':1, 'batch_size':256}

    if ds == 7:
        # WS
        hd5_in = "/media/thomas/SSD/Samples/WS2/WS2_d0.h5"
        hd5_key = "dataset_1"
        rec_prms = {
            "E0": 60.0,
            "apeture": 25.0,
            "gmax": 0.6089,
            "cbed_size": 64,
            "step_size": 0.09,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1.0,
        }
        options={'threads':1, 'batch_size':256}

    if ds == 8:
        # Au
        hd5_in = "/media/thomas/SSD/Samples/Au/Au_big_crop2.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 1.6254,
            "cbed_size": 128,
            "step_size": 0.1394,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'threads':1, 'batch_size':256}

    if ds == 9:
        # In2Se3
        hd5_in = "/media/thomas/SSD/Samples/In2Se3/default10.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 200.0,
            "apeture": 23.0,
            "gmax": 1.6254,
            "cbed_size": 128,
            "step_size": 0.0314,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'threads':1, 'batch_size':256}

    # MoS2
    if ds == 10:
        hd5_in = "/media/thomas/SSD/Samples/MSO/mos_Cs.h5"
        hd5_key = "ds_int"
        rec_prms = {
            "E0": 200.0,
            "apeture": 10.0,
            "gmax": 1.9973,
            "cbed_size": 128,
            "step_size": 0.1,
            "aberrations": [-1, 1.0],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1,
        }
        options={'threads':1, 'batch_size':256}

    if ds == 11:
        # Au 2
        hd5_in = "/media/thomas/SSD/Samples/Au/Au_NPb.h5"
        # hd5_in = "/media/thomas/SSD/Samples/Au/Au_NP.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 4.0635,
            "cbed_size": 128,
            "step_size": 0.12587,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'threads':1, 'batch_size':256}
    

    ##########################
    #       Safiyye          #
    ##########################

    if ds == 20:
        # MOF_PCN_222
        # hd5_in = "/media/thomas/SSD/Samples/MOF_PCN_222/64x64x1000x1000_cheat2_centered.h5"
        hd5_in = "/media/thomas/SSD/People/Safiyye/Thu_Sep_29_14_48_24_2022_STEM_300kV_2048x2048_3_us_1_scans_2000x2000_yx_skip3_sum6.h5"
        # hd5_in = "/media/thomas/SSD/Samples/MOF_PCN_222/64x64x1000x1000_centered.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 10.0,
            "gmax": 4.0635,
            "cbed_size": 64,
            "step_size": 0.238,
            # "step_size": 0.358,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'threads':1, 'batch_size':256}
    
    if ds == 21:
        # MOF_PCN_222
        hd5_in = "/media/thomas/SSD/People/Safiyye/Thu_Dec_15_12_58_36_2022_STEM_300kV_2048x2048_2.0_us_1_scans_alpha0_64_b1.h5"
        # hd5_in = "/media/thomas/SSD/People/Safiyye/Thu_Dec_15_12_58_36_2022_STEM_300kV_2048x2048_2.0_us_1_scans_alpha0_64_centered.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 12.0,
            "gmax": 4.0635,
            "cbed_size": 64,
            "step_size": 2.88,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1.0,
        }
        options={'threads':1, 'batch_size':256}
    
    if ds == 22:
        # MOF_PCN_222
        # hd5_in = "/media/thomas/SSD/People/Safiyye/Wed_Jan_25_15_36_30_2023_STEM_300kV_2048x2048_2.0_us_2_scans_alpha-42_yx_skip2_sum3.h5"
        hd5_in = "/media/thomas/SSD/People/Safiyye/Thu_Dec_15_12_58_36_2022_STEM_300kV_2048x2048_2.0_us_1_scans_alpha0_64_centered.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 10.0,
            "gmax": 4.0635,
            "cbed_size": 64,
            "step_size": 2.88*3,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1,
        }
        options={'threads':1, 'batch_size':256}

    ##########################
    #        Nadine          #
    ##########################

    if ds == 30:
        # hd5_in = "/media/thomas/SSD/People/Nadine/Thu_Sep_29_14_48_24_2022_STEM_300kV_2048x2048_3_us_1_scans_1000x1000_yx_skip2_sum3.h5"
        # hd5_in = "/media/thomas/SSD/People/Nadine/Tue_Oct_25_20_14_24_2022_STEM_200kV_2048x2048_1_us_3_scans_1000x1000x32x32_yx_skip2_sum3.h5"
        hd5_in = "/media/thomas/SSD/People/Nadine/Tue_Oct_25_20_14_24_2022_STEM_200kV_2048x2048_1_us_3_scans_1000x1000_yx_skip2_sum3.h5"
        # hd5_in = "/media/thomas/SSD/People/Nadine/Thu_Mar__3_15_25_48_2022_STEM_200kV_2048x2048_1_us_stepsize_0.081_scans_yx.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 200.0,
            "apeture": 13.0,
            "gmax": 4.0635,
            "cbed_size": 64,
            "step_size": 0.21*3,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 1,
        }
        options={'threads':1, 'batch_size':256}

    if ds == 31:
        hd5_in = "/media/thomas/SSD/People/Nadine/Thu_Mar__3_15_25_48_2022_STEM_200kV_2048x2048_1_us_stepsize_0.081_scans_yx_skip2_sum3.h5"
        # hd5_in = "/media/thomas/SSD/People/Nadine/Thu_Mar__3_15_25_48_2022_STEM_200kV_2048x2048_1_us_stepsize_0.081_scans_yx.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 200.0,
            "apeture": 21.0,
            "gmax": 4.0635,
            "cbed_size": 64,
            "step_size": 0.081*3,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2,
        }
        options={'threads':1, 'batch_size':256}

    if ds == 32:
        hd5_in = "/media/thomas/SSD/People/Nadine/Mon_Feb_14_14_09_50_2022_STEM_200kV_2048x2048_6_us_stepsize_0.081_scans_2000x2000_yx_skip2.h5"
        # hd5_in = "/media/thomas/SSD/People/Nadine/Thu_Mar__3_15_25_48_2022_STEM_200kV_2048x2048_1_us_stepsize_0.081_scans_yx.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 200.0,
            "apeture": 21.0,
            "gmax": 4.0635,
            "cbed_size": 64,
            "step_size": 0.081*3,
            "aberrations": [-1, 1e-3],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2,
        }
        options={'threads':1, 'batch_size':256}

    ##########################
    # Yansongs squishy stuff #
    ##########################
    if ds == 40:
        # Yansongs squishy stuff
        hd5_in = "/media/thomas/SSD/People/Yansong/yansong_64.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 21.0,
            "gmax": 4.0635,
            "cbed_size": 64,
            "step_size": 0.21484375,
            "aberrations": [0, 0],
            "probe_estimation_method": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'threads':1, 'batch_size':256}
    

    dose_cbed = args["dose"]*(rec_prms['step_size']*args['skip'])**2
    if dose_cbed < 1:
        dose_cbed = None
    folder, file_name = os.path.split(hd5_in)
    file_name = os.path.splitext(os.path.basename(file_name))[0]
    out_path = os.path.join(folder,file_name + "_s_" + str(args["skip"]) + '_d_' + str(args["dose"]) + '_airpi.mat')
    start = time.time()
    obj = reconstruct_ds(hd5_in, hd5_key, prms_net, rec_prms, skip=int(args["skip"]), dose=dose_cbed, options=options)
    end = time.time()
    print(end - start)

    rec_prms['skip'] = args["skip"]
    rec_prms['dose'] = args["dose"]
    rec_prms['model'] = args["model"]
    sio.savemat(out_path, {"obj": obj, "prms": rec_prms})
    quit()