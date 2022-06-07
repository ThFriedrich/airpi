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
    _, prms_net = load_obj(os.path.join(cp_path, "hyperparameters.txt"))
    prms_net["cp_path"] = cp_path
    prms_net["predict"] = True
    return prms_net


if __name__ == "__main__":
    os.system("clear")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", type=int, default=0, help="Dose")
    parser.add_argument("--ds", type=int, default=0, help="Dataset")
    parser.add_argument("--step", type=int, default=1, help="Step skip")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU")
    args = vars(parser.parse_args())
    dose = int(args["dose"])
    ds = int(args["ds"])
    if dose == 0:
        dose = None

    # cp_path = 'airpi/ap_training/Ckp/Training/CNET_U_32_3_4'
    # cp_path = 'airpi/ap_training/Ckp/Training/U_test'
    # cp_path = 'airpi/ap_training/Ckp/Training/BR_64_2_low_amp'
    cp_path = 'airpi/ap_training/Ckp/Training/BR_64_2'
    # cp_path = 'airpi/ap_training/Ckp/Training/CNET_64'
    # cp_path = 'airpi/ap_training/Ckp/Training/BRANCHED_RES_V_ADAM_CAT2b'
    # cp_path = "airpi/ap_training/Ckp/Training/BRANCHED_RES_V_ADAM_CAT3"
    # cp_path = "airpi/ap_training/Ckp/Training/BR7_4_3_32"
    # cp_path = "airpi/ap_training/Ckp/Training/BRANCHED_RES_V_ADAM_CAT3_ld"
    # cp_path = "airpi/ap_training/Ckp/Training/BRANCHED_RES_V_ADAM_CAT3_rev2_ld3"
    bfm_type = "avrg"  # 'reco: pre-recorded cbed, gene: parameter generated cbed, avrg: use PACBED, None: No masking')
    # bfm_type = 'gene'  # 'reco: pre-recorded cbed, gene: parameter generated cbed, avrg: use PACBED, None: No masking')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])

    from airpi.ap_reconstruction.reconstruction_functions import (
        retrieve_phase_from_generator,
    )
    from airpi.ap_reconstruction.airpi_dataset import airpi_dataset
    from airpi.ap_architectures.utils import load_obj

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
            "bfm_type": 'gene',
            "oversample": 1,
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
            "bfm_type": 'gene',
            "oversample": 1,
        }

    if ds == 2:
        # STO
        hd5_in = "/media/thomas/SSD/Samples/STO/hole.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 1.6671,
            "cbed_size": 128,
            "step_size": 0.1818,
            "aberrations": [-1, 1e-3],
            "bfm_type": 'gene',
            "oversample": 1.0,
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
            "bfm_type": 'avrg',
            "oversample": 1.0,
        }
    if ds == 4:
        # Au
        hd5_in = "/media/data/Samples/Au_big_crop2.h5"
        # hd5_in = "/media/thomas/SSD/Samples/Au/Au_big.h5"
        hd5_key = "ds"
        rec_prms = {
            "E0": 300.0,
            "apeture": 20.0,
            "gmax": 1.6254,
            "cbed_size": 128,
            "step_size": 0.1394,
            "aberrations": [-1, 1e-3],
            "bfm_type": "avrg",
            "oversample": 1.0,
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
            "bfm_type": "avrg",
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
            "bfm_type": bfm_type,
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
            "bfm_type": "gene",
            "oversample": 1.0,
        }

    if ds == 8:
        # Z
        hd5_in = "/media/thomas/SSD/Samples/Z/Z/db_h5_Z_sim_6.h5"
        hd5_key = "amp"
        rec_prms = {
            "E0": 300.0,
            "apeture": 30.0,
            "gmax": 12.6984,
            "cbed_size": 128,
            "step_size": 1.2,
            "aberrations": [-1, 1e-3],
            "bfm_type": "gene",
            "oversample": 1.0,
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
            "bfm_type": 'avrg',
            "oversample": 1.0,
        }

    reconstruct_ds(hd5_in, hd5_key, prms_net, rec_prms, skip=int(args["step"]), dose=dose, save=True)
    print("done")
    quit()