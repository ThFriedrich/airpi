import sys
import h5py
import tensorflow as tf
from numpy import logspace

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