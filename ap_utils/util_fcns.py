from io import BytesIO
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import ix_, zeros, logspace
from pickle import dump, load
import os
import sys
import h5py

from ap_architectures.utils import pred_dict, tf_pow01, true_dict
from ap_reconstruction.colormaps import parula

def fcn_decode_tfrecords(record):
    '''Decoding/Preprocessing of TFRecord Dataset'''
    features = {
        'features': tf.io.FixedLenFeature([1], tf.string),
        'meta': tf.io.FixedLenFeature([19], tf.float32),
        'labels_k': tf.io.FixedLenFeature([1], tf.string),
        'labels_r': tf.io.FixedLenFeature([1], tf.string),
        'probe_r': tf.io.FixedLenFeature([1], tf.string)
    }

    parsed_features = tf.io.parse_single_example(record, features)
    return parsed_features

# Training run setup, argument passing, etc.
def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, "gettrace", lambda: None)
    return gettrace() is not None


class Parameters:
    """Class to avoid global variables"""

    def __init__(self, X_SHAPE):
        self.X_SHAPE = X_SHAPE
        self.n_train = 0
        self.n_val = 0
        self.ds_order = [1, 2, 0, 3]
        self.dose = tf.constant(logspace(3, 10, num=2048) * 5.0, dtype=tf.float32)
        self.model = "dbg"
        self.model_loaded = False
        self.debug = debugger_is_active()

    def __tuple2array__(self, tp1, tp2):
        ar = list(tp1)
        ar.append(tp2)
        return [ar[i] for i in self.ds_order]

    def get_ds_size(self,path):
        if path.endswith(".h5"):
            with h5py.File(path, "r") as hf:
                feat_shape = hf["features"][0].shape
                num_feat = len(hf["features"])
        elif path.endswith(".tfrecord"):
            num_feat = 0
            ds = tf.data.TFRecordDataset(path,num_parallel_reads=8).map(fcn_decode_tfrecords)
            num_feat += sum(1 for _ in ds)
            feat_shape = [9,64,64]
            # num_feat = 742688
        return num_feat, feat_shape 

    def get_database_sizes(self, prms):
        """Read global variables"""
        for path in prms["train_path"]:
            # num_feat, feat_shape = self.get_ds_size(path)
            feat_shape = [9,64,64]
            num_feat = 742688
            self.n_train += num_feat
            self.X_SHAPE = self.__tuple2array__(feat_shape, num_feat)
        for path in prms["val_path"]:
            self.n_val += self.get_ds_size(path)[0]
        self.model = prms["model_name"]

        return self



PRM = Parameters(0)

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
        "scale_cbeds": True,
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
    log_path = os.path.join(cwd, "Logs", "Training_5",model_name)

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


# Tensorboard image logging stuff
def fcn_plot_to_image(figure):
    """Create images from Plots"""
    # Save the plot to a PNG in memory.
    buf = BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image

def fcn_plot_example(obj,epoch=0,name='default'):
    """Plot example of a full reconstruction"""
    fig = plt.figure(figsize=(7, 7))
    img = plt.imshow(obj, cmap=parula)
    plt.axis('off')
    plt.colorbar(img,fraction=0.046, pad=0.04)
    path = os.path.join("Images", PRM.model, "Epoch_" + str(epoch))

    os.makedirs(path, exist_ok=True)
    plt.savefig(
        os.path.join(path, PRM.model + "_" + name + "_" + str(epoch+1) + ".png")
    )
    im = fcn_plot_to_image(fig)
    return im


def fcn_plot_nn_in_out(feat, pred, lab=None, epoch=0, suffix="0"):
    """
    Plot input and output of neural network for the test dataset.
    Input:
        feat: features of test dataset
        pred: predictions of test dataset
        lab: labels of test dataset
        epoch: epoch of training
        suffix: suffix of plot
    """

    def format_axes(fig):
        for ax in fig.axes:
            ax.set_axis_off()
            plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

    # Create figure
    pred_r, pred_k = pred_dict(pred)
    lab_r, lab_k = true_dict(lab)
    pred_r["obj"] = tf.math.angle(pred_r["wv"]/lab_r["probe"])
    weight_r = tf.math.abs(lab_r["probe"])
    weight_r /= tf.math.reduce_sum(weight_r)

    fig = plt.figure(figsize=(15, 4.75))
    gs = GridSpec(4, 13, figure=fig)

    px_s = feat.shape[1]
    n_im = 3

    px_c = px_s * n_im
    collage = zeros((px_c, px_c))


    for iy in range(n_im):
        py_rng = range(px_s * iy, px_s * (iy + 1))
        for ix in range(n_im):
            px_rng = range(px_s * ix, px_s * (ix + 1))
            collage[ix_(py_rng, px_rng)] = feat[..., iy * n_im + ix]
    collage = tf_pow01(collage)

    # Inputs
    ax1 = fig.add_subplot(gs[0:3, 0:3])
    ax1.imshow(collage)
    ax1.set_title("Input")

    ax2 = fig.add_subplot(gs[3:, 0:3])
    ax2.imshow(feat[..., 9])
    ax2.set_title("Mask k")

    # Label
    ax3 = fig.add_subplot(gs[0:2, 3:5])
    ax3.imshow(lab_k["amp_sc"])
    ax3.set_title("True k Amp")

    ax4 = fig.add_subplot(gs[0:2, 5:7])
    ax4.imshow(lab_k["phase"])
    ax4.set_title("True k Phase")

    ax5 = fig.add_subplot(gs[0:2, 7:9])
    ax5.imshow(lab_r["amp_sc"])
    ax5.set_title("True r Amp")

    ax6 = fig.add_subplot(gs[0:2, 9:11])
    ax6.imshow(lab_r["phase"])
    ax6.set_title("True r Phase")

    ax7 = fig.add_subplot(gs[0:2, 11:13])
    ax7.imshow(lab_r["obj"]*weight_r)
    ax7.set_title("True Object")

    # Predictions
    ax8 = fig.add_subplot(gs[2:, 3:5])
    ax8.imshow(pred_k["amp_sc"])
    ax8.set_title("Pred k Amp")

    ax9 = fig.add_subplot(gs[2:, 5:7])
    ax9.imshow(pred_k["phase"])
    ax9.set_title("Pred k Phase")

    ax10 = fig.add_subplot(gs[2:, 7:9])
    ax10.imshow(pred_r["amp_sc"])
    ax10.set_title("Pred r Amp")

    ax11 = fig.add_subplot(gs[2:, 9:11])
    ax11.imshow(pred_r["phase"])
    ax11.set_title("Pred r Phase")

    ax12 = fig.add_subplot(gs[2:, 11:13])
    ax12.imshow(pred_r["obj"]*weight_r)
    ax12.set_title("Pred Object")

    format_axes(fig)
    plt.tight_layout()

    path = os.path.join("Images", PRM.model, "Epoch_" + str(epoch))
    os.makedirs(path, exist_ok=True)
    plt.savefig(
        os.path.join(path, PRM.model + "e_" + str(epoch+1) + "s_" + suffix + ".png")
    )

    im = fcn_plot_to_image(fig)
    return im
