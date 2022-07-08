import os
import argparse
import silence_tensorflow.auto
import nvidia_smi


def get_gpu_memory(n_gpu):

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(n_gpu)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    nvidia_smi.nvmlShutdown()

    return info


def reconstruct_ds(ds_path, key, prms_net, rec_prms, skip=1, dose=None, save=False):
    (file, _) = os.path.splitext(ds_path)
    rec_prms['step_size'] *= skip
    nn_name = prms_net['cp_path'].split("/")[-1]
    if save:
        ds_ews_path = file + "_ews_" + nn_name + "_step_" + str((skip)) + "_dose_" + str((dose)) + ".h5"
    else:
        ds_ews_path = None
    ds_class = airpi_dataset(rec_prms, ds_path, key, dose, step=skip, in_memory=False)
    retrieve_phase_from_generator(ds_class, prms_net, rec_prms, ds_ews_path)


def get_model_ckp(cp_path):
    hp_file = os.path.join(cp_path, "hyperparameters.pickle")
    _, prms_net = load_hparams(hp_file)
    prms_net["cp_path"] = cp_path
    prms_net["deploy"] = True
    return prms_net


if __name__ == "__main__":
    os.system("clear")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", type=int, default=1000, help="Dose")
    parser.add_argument("--ds", type=int, default=1, help="Dataset")
    parser.add_argument("--step", type=int, default=4, help="Step skip")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU")
    # parser.add_argument("--model", type=str, default='CNET_16_D4_e', help="GPU")
    # parser.add_argument("--model", type=str, default='CNET_32_D4_b', help="GPU")
    parser.add_argument("--model", type=str, default='CNET_32_D3_f', help="GPU")
    # parser.add_argument("--model", type=str, default='V_32_D3_RLA_d_e2_skip', help="GPU")
    # parser.add_argument("--model", type=str, default='CNET_32_D5b', help="GPU")
    parser.add_argument("--ap_fcn", type=str, default='avrg', help="Aperture function estimation, gene: parameter generated, avrg: use PACBED")
    args = vars(parser.parse_args())
    dose = int(args["dose"])
    ds = int(args["ds"])
    if dose == 0:
        dose = None

    cp_path = 'Ckp/Training/' + args["model"]
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])

    from ap_reconstruction.reconstruction_functions import (
        retrieve_phase_from_generator,
    )
    from ap_reconstruction.airpi_dataset import airpi_dataset
    from ap_utils.util_fcns import load_hparams
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

    # MoS2
    if ds == 1:
        hd5_in = "/media/thomas/SSD/Samples/MSO/airpi_sto.h5"
        hd5_key = "ds_int"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 4.5714,
            "cbed_size": 128,
            "step_size": 0.05,
            "aberrations": [14.0312, 1e-3],
            "bfm_type": args["ap_fcn"],
            "oversample": 2.0,
        }

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
            "step_size": 0.1818*2,
            "aberrations": [-1, 1e-3],
            "bfm_type": args["ap_fcn"],
            "oversample": 2.0,
        }
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
    if ds == 4:
        # Au
        # hd5_in = "/media/data/Samples/Au_big_crop2.h5"
        hd5_in = "/media/thomas/SSD/Samples/Au/Au_big.h5"
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
            "oversample": 1.0,
        }

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

    if ds == 7:
        # Au 2
        hd5_in = "/media/data/Samples/Au_NPb.h5"
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

    if ds == 8:
        # Z
        hd5_in = "/media/thomas/SSD/Samples/Z/db_h5_6_sim.h5"
        hd5_key = "amp"
        rec_prms = {
            "E0": 200.0,
            "apeture": 20.0,
            "gmax": 0.7975,
            "cbed_size": 64,
            "step_size": 0.2,
            "aberrations": [-1, 1e-3],
            "bfm_type": "gene",
            "oversample": 2.0,
            "order":['kx','ky','rx','ry']
        }

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

    reconstruct_ds(hd5_in, hd5_key, prms_net, rec_prms, skip=int(args["step"]), dose=dose, save=False)
    print("done")
    quit()