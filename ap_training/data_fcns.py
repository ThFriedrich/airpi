"""Module containing Functions for Tensorflow Training and Grid Search"""
import numpy as np
import tensorflow as tf
from ap_architectures.utils import tf_binary_mask, tf_fft2d_A_p, tf_pi
import h5py

from ap_utils.util_fcns import PRM


# @tf.function
def cast_float32(x: tf.Tensor, meta: tf.Tensor, shape: tf.TensorShape) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    x = x * tf.broadcast_to(tf.expand_dims(meta, 0), shape) / 65536

    return x


# @tf.function
def cast_wave_float32(x: tf.Tensor, sc: tf.Tensor) -> tf.Tensor:
    # Amp & Phase
    sh_s = tf.TensorShape((64, 64))
    y_amp = cast_float32(x[:, :, 1], sc[1], sh_s)
    y_phase = cast_float32(x[:, :, 0], sc[0], sh_s) - tf_pi
    y = tf.stack([y_amp, y_phase], axis=-1)

    return y


class generator:
    """Generator that keeps the HDF5-DB open and reads chunk-by-chunk"""

    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, "r") as hf:
            for ds in range(len(hf["features"])):
                meta = hf["meta"][ds]
                features = hf["features"][ds]
                labels_k = hf["labels_k"][ds]
                labels_r = hf["labels_r"][ds]
                probe_r = hf["probe_r"][ds]

                yield features, labels_k, labels_r, probe_r, meta


def get_h5_ds(hdf5_path):
    ds = tf.data.Dataset.from_generator(
        generator(hdf5_path),
        output_types=(tf.uint16, tf.uint16, tf.uint16, tf.uint16, tf.float32),
        output_shapes=(
            tf.TensorShape([9, 64, 64]),
            tf.TensorShape([2, 64, 64]),
            tf.TensorShape([2, 64, 64]),
            tf.TensorShape([2, 64, 64]),
            tf.TensorShape([19]),
        ),
    )

    return ds


@tf.function
def fcn_decode_h5(fea, lab_k, lab_r, prob_r, met):

    # [E_0, cond_lens_outer_aper_ang, g_max, step_size] , [scale_x], [scale_yk], [scale_yr], [scale_pr]
    prms, sc_x, sc_yk, sc_yr, sc_pr = tf.split(met, [4, 9, 2, 2, 2])

    # Needs transposing cause Matlab is doing column-major indexing
    fea = tf.transpose(fea, [2, 1, 0])
    lab_k = tf.transpose(lab_k, [2, 1, 0])
    lab_r = tf.transpose(lab_r, [2, 1, 0])
    prob_r = tf.transpose(prob_r, [2, 1, 0])

    # 3x3 cbed-kernel
    sh_x = tf.TensorShape((64, 64, PRM.X_SHAPE[2]))
    x = cast_float32(fea, sc_x, sh_x)

    # Label and probe
    lab_k = cast_wave_float32(lab_k, sc_yk)
    lab_r = cast_wave_float32(lab_r, sc_yr)
    prob_r = cast_wave_float32(prob_r, sc_pr)

    return x, lab_k, lab_r, prob_r, prms


@tf.function
def fcn_rnd_no_sample(x, yk, probe, p):
    rnd = tf.random.uniform([])
    x = tf.cond(
        tf.greater(p, rnd),
        lambda: tf.tile(probe[..., 0, tf.newaxis] ** 2, [1, 1, 9]),
        lambda: x,
    )
    yk = tf.cond(tf.greater(p, rnd), lambda: probe, lambda: yk)

    return x, yk

@tf.function
def add_binary_mask(x:tf.Tensor, prob_r:tf.Tensor) -> tf.Tensor:
    prob_k = tf_fft2d_A_p(prob_r)
    b_msk = tf_binary_mask(prob_k[...,0])
    x = tf.concat([x, b_msk[...,tf.newaxis]], -1)

    return x

@tf.function
def fcn_decode_train(fea, lab_k, lab_r, prob_r, met):
    """Decoding/Preprocessing of HDF5 Dataset for Training (random Poisson)"""
    x, lab_k, lab_r, prob_r, _ = fcn_decode_h5(fea, lab_k, lab_r, prob_r, met)

    # Random no-sample
    x, lab_k = fcn_rnd_no_sample(x, lab_k, tf_fft2d_A_p(prob_r), 0.03)

    # Add Poisson Noise for random dose
    rnd = tf.random.uniform([1], minval=0, maxval=len(PRM.dose), dtype=tf.int32)
    d = tf.gather(PRM.dose, rnd)
    x = tf.squeeze(tf.random.poisson([1], x * d))

    if PRM.scale_cbeds:
        x = fcn_weight_cbeds(x, met[3])

    x = add_binary_mask(x, prob_r)

    y = tf.concat([lab_k, lab_r, prob_r], -1)

    return x, y

@tf.function
def fcn_decode_val(fea, lab_k, lab_r, prob_r, met):
    """Decoding/Preprocessing of HDF5 Dataset for Validation (random Poisson, doses same)"""
    x, lab_k, lab_r, prob_r, prms = fcn_decode_h5(fea, lab_k, lab_r, prob_r, met)

    # Generate a seed based on sample meta-data
    prm_prod = tf.reduce_prod(prms[0:2])
    s1 = tf.cast(tf.round(prm_prod), tf.int64)
    s2 = tf.cast(tf.round(prm_prod + prms[2] * 100), tf.int64)

    # Apply noise from seed
    rnd = tf.random.stateless_uniform(
        [1], seed=(s1, s2), minval=0, maxval=len(PRM.dose), dtype=tf.int32
    )
    d = tf.gather(PRM.dose, rnd)
    x = tf.squeeze(tf.random.poisson([1], x * d, seed=13))

    if PRM.scale_cbeds:
        x = fcn_weight_cbeds(x, met[3])

    x = add_binary_mask(x, prob_r)
    y = tf.concat([lab_k, lab_r, prob_r], -1)

    return x, y


@tf.function
def fcn_weight_cbeds(cbeds, step_size):
    """Weighting of input CBEDs, according to step size"""
    d1 = (1 / step_size) / 50
    d2 = (1 / tf.math.sqrt(2 * step_size**2)) / 50
    ind_1 = [[1], [3], [5], [7]]
    ind_2 = [[0], [2], [6], [8]]

    cbeds = tf.transpose(cbeds,[2,0,1])
    cbeds = tf.tensor_scatter_nd_update(cbeds, ind_1, tf.gather_nd(cbeds,ind_1)*d1)
    cbeds = tf.tensor_scatter_nd_update(cbeds, ind_2, tf.gather_nd(cbeds,ind_1)*d2)
    cbeds = tf.transpose(cbeds,[1,2,0])

    return cbeds


def datasetPipeline(filepaths, is_training, prms):
    """HDF5 Dataset-Pipeline"""
    datasets = list()
    for path in filepaths:
        tf.print(path)
        datasets.append(get_h5_ds(path))

    if is_training:
        dataset = tf.data.experimental.sample_from_datasets(datasets)
        # Add the augmentations to the dataset
        dataset = dataset.map(
            fcn_decode_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        # dataset = dataset.shuffle(4096)
        dataset = dataset.repeat(prms["epochs"])
    else:
        dataset = tf.data.experimental.sample_from_datasets(datasets, seed=1310)
        dataset = dataset.map(
            fcn_decode_val, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Set the batchsize
    dataset = dataset.batch(prms["batch_size"], drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def getDatasets(prms):
    if PRM.debug:
        n_train_data = 32
    else:
        n_train_data = PRM.n_train

    validation_steps = (
        np.min([PRM.n_val, int(n_train_data * 0.05)]) // prms["batch_size"]
    )

    steps_per_epoch = n_train_data // prms["batch_size"]

    prms.update(
        {"steps_per_epoch": steps_per_epoch, "validation_steps": validation_steps}
    )

    PRM.scale_cbeds = prms["scale_cbeds"]

    training_ds = datasetPipeline(prms["train_path"], True, prms)
    validation_ds = datasetPipeline(prms["val_path"], False, prms)

    return (training_ds, validation_ds)


def getTestDataset(filename, batch_size):
    """Return Dataset for Visualisation"""
    prms = {
        "train_path": filename,
        "epochs": None,
        "batch_size": batch_size
    }
    dataset = datasetPipeline(filename, False, prms)

    return dataset
