import os
import argparse
import silence_tensorflow.auto
import scipy.io as sio


def reconstruct_ds(ds_path, key, prms_net, rec_prms, options, skip=1, dose=None):
    (file, _) = os.path.splitext(ds_path)
    rec_prms['step_size'] *= skip
    nn_name = prms_net['cp_path'].split("/")[-1]
    if options['ew_ds_path'] is not None:
        options['ew_ds_path'] = file + "_ews_" + nn_name + "_step_" + str((skip)) + "_dose_" + str((dose)) + ".h5"
    else:
        options['ew_ds_path'] = None
    ds_class = airpi_dataset(rec_prms, ds_path, key, dose, step=skip, in_memory=False)
    return retrieve_phase_from_generator(ds_class, prms_net, options)


def get_model_ckp(cp_path):
    hp_file = os.path.join(cp_path, "hyperparameters.pickle")
    _, prms_net = load_hparams(hp_file)
    prms_net["cp_path"] = cp_path
    return prms_net


if __name__ == "__main__":
    os.system("clear")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", type=int, default=40, help="Dose")
    parser.add_argument("--ds", type=int, default=1, help="Dataset")
    parser.add_argument("--step", type=int, default=4, help="Step skip")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU")
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

    from tensorflow import config as tf_config
    tf_config.optimizer.set_jit("autoclustering")

    from ap_reconstruction.reconstruction_functions import (
        retrieve_phase_from_generator,
    )
    from ap_reconstruction.airpi_dataset import airpi_dataset
    from ap_utils.file_ops import load_hparams
    from ap_utils.globals import debugger_is_active
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
            "bfm_type": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'b_offset_correction':False, 'threads':4, 'ew_ds_path':None, 'batch_size':64}
    
    if ds == 12:
        hd5_in = "/media/thomas/SSD/Samples/graphene/gra_tim.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 60.0,
            "apeture": 34,
            "gmax": 1.8632454,
            "cbed_size": 64,
            "step_size": 0.04,
            "aberrations": [0, 0],
            "bfm_type": args["ap_fcn"],
            "oversample": 1.0,
            "order": ['ry','rx','kx','ky']
        }
        options={'b_offset_correction':True, 'threads':4, 'ew_ds_path':None}

    if ds == 13:
        hd5_in = "/media/thomas/SSD/Samples/STO/sto_sro.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20,
            "gmax": 1.6253973,
            "cbed_size": 64,
            "step_size": 0.43,
            "aberrations": [-1, 1e-3],
            "bfm_type": args["ap_fcn"],
            "oversample": 2.0,
            "order": ['ry','rx','kx','ky']
        }
        options={'b_offset_correction':False, 'threads':5, 'ew_ds_path':None}
    
    if ds == 14:
        hd5_in = "/media/thomas/SSD/Samples/MgO/MgO.h5"
        hd5_key = "ds_cbed"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20,
            "gmax": 1.6253973,
            "cbed_size": 64,
            "step_size": 0.05,
            "aberrations": [-1, 1e-3],
            "bfm_type": args["ap_fcn"],
            "oversample":2.0,
            "order": ['rx','ry','kx','ky']
        }
        options={'b_offset_correction':False, 'threads':6, 'ew_ds_path':None, 'batch_size':64}

    # MoS2ew_ds_path
    if ds == 1:
        hd5_in = "/media/thomas/SSD/Samples/MSO/airpi_sto.h5"
        hd5_key = "ds_int"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 4.5714,
            "cbed_size": 128,
            "step_size": 0.05,
            "aberrations": [-1, 1e-3],
            # "aberrations": [14.0312, 1e-3],
            "bfm_type": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}

    if ds == 2:
        # STO
        hd5_in = "/media/thomas/SSD/Samples/STO/hole_preprocessed_cropped_2.h5"
        # hd5_in = "/media/thomas/SSD/Samples/STO/hole.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 1.6671,
            "cbed_size": 64,
            "step_size": 0.1818,
            "aberrations": [-1, 1e-3],
            "bfm_type": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}

    if ds == 3:
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
            "bfm_type": args["ap_fcn"],
            "oversample": 1.0,
        }
        options={'b_offset_correction':False, 'threads':8, 'ew_ds_path':None}

    if ds == 4:
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
            "bfm_type": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None, 'batch_size':64}

    if ds == 5:
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
            "bfm_type": args["ap_fcn"],
            "oversample": 2.0,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None, 'batch_size':8}

    # MoS2
    if ds == 6:
        hd5_in = "/media/thomas/SSD/Samples/MSO/mos_Cs.h5"
        hd5_key = "ds_int"
        rec_prms = {
            "E0": 200.0,
            "apeture": 10.0,
            "gmax": 1.9973,
            "cbed_size": 128,
            "step_size": 0.1,
            "aberrations": [-1, 1.0],
            "bfm_type": args["ap_fcn"],
            "oversample": 1,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}

    if ds == 7:
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
            "bfm_type": args["ap_fcn"],
            "oversample": 1.0,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}

    if ds == 8:
        # Z
        hd5_in = "/media/thomas/SSD/Samples/MoS2/mos2_cornell.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 80.0,
            "apeture": 21.4,
            "gmax": 0.7975,
            "cbed_size": 64,
            "step_size": 0.2,
            "aberrations": [-1, 1e-3],
            "bfm_type":  args["ap_fcn"],
            "oversample": 2.0
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}

    if ds == 9:
        # WS2
        hd5_in = "/media/thomas/SSD/Samples/WS/ws2_preprocessed.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 60.0,
            "apeture": 25.0,
            "gmax": 0.802,
            "cbed_size": 64,
            "step_size": 0.09,
            "aberrations": [-1, 1e-3],
            "bfm_type": args["ap_fcn"],
            "oversample": 1.0,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}

    if ds == 10:
        # WS2
        hd5_in = "/media/thomas/SSD/Samples/number.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 60.0,
            "apeture": 25.0,
            "gmax": 0.802,
            "cbed_size": 64,
            "step_size": 0.09,
            "aberrations": [-1, 1e-3],
            "bfm_type": args["ap_fcn"],
            "oversample": 1.0,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}

    if ds == 113:
        # Z
        hd5_in = "/media/thomas/SSD/Samples/Z64/db_h5_47_sim_64.h5"
        hd5_key = "int"
        rec_prms = {
            "E0": 200.0,
            "apeture": 20.0,
            "gmax": 2.3924,
            "cbed_size": 64,
            "step_size": 0.2,
            "aberrations": [-1, 1e-3],
            "bfm_type": 'avrg',
            "oversample": 1.0,
        }
        options={'b_offset_correction':False, 'threads':1, 'ew_ds_path':None}

    file_name = os.path.splitext(os.path.basename(hd5_in))[0]
    out_path = os.path.join('Scripts/rev2',file_name + "_s_" + str(args["step"]) + '_d_' + str(args["dose"]) + '.mat')
    obj = reconstruct_ds(hd5_in, hd5_key, prms_net, rec_prms, skip=int(args["step"]), dose=dose, options=options)
    
    rec_prms['skip'] = args["step"]
    rec_prms['dose'] = args["dose"]
    rec_prms['offset_correction'] = options['b_offset_correction']
    rec_prms['model'] = args["model"]

    sio.savemat(out_path, {"obj": obj, "prms": rec_prms})
    quit()