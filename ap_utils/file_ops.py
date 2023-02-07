import tensorflow as tf
from numpy import logspace
from pickle import dump, load
import os
from ap_utils.globals import PRM



def get_default_prms(hp_path, log_path):
    # Construct default dictionaries for hyperparameters
    prms = {
        "model_name": "dbg",
        # Data
        "train_path": absoluteFilePaths("/media/thomas/SSD/Data_64/Training"),
        "val_path": absoluteFilePaths("/media/thomas/SSD/Data_64/Validation"),
        "test_path": absoluteFilePaths("/media/thomas/SSD/Data_64/Test"),
        "sample_dir": "/media/thomas/SSD/Samples/",
        "log_path": log_path,
        "cp_path": hp_path,
        # Learning Rate schedule parameters
        "learning_rate": 1e-6,
        "learning_rate_0":1e-6,
        "epochs": 300,
        "ep": 0,
        "epochs_cycle_1": 50,
        "epochs_cycle": 20,
        "epochs_ramp": 5,
        "warmup": False,
        "cooldown": True,
        "lr_fact": 0.75,
        # Optimizer, Model Parameters
        "loss_prms_r": [1.0, 1.0, 1.0, 25.0, 2.0, 1.0, 0.1],
        "loss_prms_k":[2.0, 4.0, 2.0, 4.0, 1.0, 1.0e-3, 1.0],
        "batch_size": 32,
        "scale_cbeds": False,
        "dose": [1, 7, 5]
    }

    prms_net = {
        "arch": "UNET",
        "branching": 0,
        "kernel": [3, 3],
        "normalization": 5,     # 0 = None, 1 = Instance, 2 = Layer else Batch
        "activation": 5,        # 0 = None, 1 = LeakyReLU, 2 = ELU, 3 = SWISH , 4=Sigmoid else ReLu
        "filters": 16,
        "depth": 3,
        "stack_n": 3,
        "type":"V",
        "w_regul": None,
        "a_regul": None,
        "dropout": None,
        "global_skip": True,
        "global_cat": False,
    }
    return prms, prms_net

def manage_args(args):
    """
    Get args from parser to alter default paramerters for training and CNN architecture.
    Also loads hyperparameters from file if model checkpoint is available and continues training from there.
    Only parameters explicitly passed by user are changed.
    This function also sets the global variable PRM.
    """

    # Default arguments
    # Weight order:
    # r: [px_p, px_as, px_pw, px_int, obj]
    # k: [px_p, px_as, px_pw, px_int, int_tot, df_ratio]

    # load hyperparameters from file or delete checkpoint if available
    if args["model_name"] is not None:
        model_name = args["model_name"]
    else:
        model_name = "dbg"

    cwd = os.path.abspath(os.path.curdir)
    hp_path = os.path.join(cwd, "Ckp", "Training", model_name)
    log_path = os.path.join(cwd, "Logs", "Training",model_name)

    prms, prms_net = load_hyperparameters(hp_path, log_path)

    # Assign default values to arguments that are not explicitly given
    for key in args:
        if args[key] is not None:
            if key in prms:
                prms[key] = args[key]
            if key in prms_net:
                prms_net[key] = args[key]
            if key == "db_path":
                prms.update(
                    {
                        "train_path": absoluteFilePaths(
                            args["db_path"] + "/Training"
                        ),
                        "val_path": absoluteFilePaths(
                            args["db_path"] + "/Validation"
                        ),
                        "test_path": absoluteFilePaths(args["db_path"] + "/Test"),
                    }
                )
            if key == "lr":
                prms.update(
                    {
                        "learning_rate": args["lr"],
                        "learning_rate_0": args["lr"],
                    }
                )
            if key == "bs":
                prms.update({"batch_size": args["bs"]})
    if "dose" not in prms:
        prms["dose"] = [1, 10, 5]
    # Set global variable PRM
    PRM.dose = tf.constant(
        logspace(prms["dose"][0], prms["dose"][1], num=2048) * prms["dose"][2],
        dtype=tf.float32,
    )

    PRM.scale_cbeds = prms["scale_cbeds"]
    PRM.get_database_sizes(prms)

    # Save hyperparameters to file
    save_hparams([prms, prms_net], os.path.join(hp_path,"hyperparameters.pickle"))
    
    return prms, prms_net


# Helper functions for file loading/saving
def save_hparams(obj, name):
    """
    Save Hyperparameters to file
    Input:
        obj: list of dictionaries [prms, prms_net]
        name: path to file
    """
    with open(name, "wb") as f:
        dump(obj, f, 0)


def load_hparams(name):
    """
    Load Hyperparameters from file
    Input:
        name: path to file
    Output:
        obj: list of dictionaries [prms, prms_net]
    """
    with open(name, "rb") as f:
        prms, prms_net = load(f)
        return prms, prms_net


def load_hyperparameters(cp_path, log_path='.'):
    """
    Check if checkpoint exists then prompt user to load or delete checkpoint.
    Input:
        cp_path: path to checkpoint
    Output:
        prms: dictionary of training hyperparameters
        prms_net: dictionary of network hyperparameters
        Both dictionaries are empty if no checkpoint was loaded.
    """
    hp_file = os.path.join(cp_path, "hyperparameters.pickle")

    prms = []
    prms_net = []
    if os.path.exists(hp_file):
        cnt = input(
            "Directory exists already. Do you want to delete(d) or continue(c) training session? (d/c) "
        )
        if cnt == "c":
            prms, prms_net = load_hparams(hp_file)
            prms['log_path'] = log_path
            prms['cp_path'] = cp_path
        elif cnt == "d":
            from shutil import rmtree
            rmtree(cp_path, ignore_errors=True)
            if os.path.exists(log_path):
                rmtree(log_path, ignore_errors=True)
            os.makedirs(cp_path)
            prms, prms_net = get_default_prms(cp_path, log_path)
    else:
        os.makedirs(cp_path, exist_ok=True)
        prms, prms_net = get_default_prms(cp_path, log_path)
    return prms, prms_net


def absoluteFilePaths(directory):
    """
    Get absolute file paths of all files in a directory
    and filter for only .h5 files
    Input:
        directory: path to directory
    Output:
        file_paths: list of absolute file paths
    """
    files = list()
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".tfrecord"):
                files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files
