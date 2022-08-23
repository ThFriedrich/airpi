"""Module containing Functions for Tensorflow Training and Grid Search"""
import numpy as np
import tensorflow as tf
from ap_utils.functions import tf_binary_mask, tf_binary_mask_r, tf_cast_complex, tf_fft2d_A_p, tf_ifft2d_A_p, tf_normalise_to_one_amp, tf_normalise_to_one_complex, tf_pi, tf_probe_function, tf_probe_k, tf_complex_interpolation
from ap_utils.globals import PRM

# @tf.function
def cast_float32(x: tf.Tensor, meta: tf.Tensor, shape: tf.TensorShape) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    x = x * tf.broadcast_to(tf.expand_dims(meta, 0), shape) / 65536

    return x


@tf.function
def cast_wave_float32(x: tf.Tensor, sc: tf.Tensor) -> tf.Tensor:
    # Amp & Phase
    sh_s = tf.TensorShape((64, 64))
    y_amp = cast_float32(x[:, :, 1], sc[1], sh_s)
    y_phase = cast_float32(x[:, :, 0], sc[0], sh_s) - tf_pi
    y = tf.stack([y_amp, y_phase], axis=-1)

    return y


@tf.function
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

    # Bring the pattern back in shape
    fea = tf.reshape(tf.io.decode_raw(
        parsed_features['features'], tf.uint16), [9, 64, 64])
    lab_k = tf.reshape(tf.io.decode_raw(
        parsed_features['labels_k'], tf.uint16), [2, 64, 64])
    lab_r = tf.reshape(tf.io.decode_raw(
        parsed_features['labels_r'], tf.uint16), [2, 64, 64])
    prob_r = tf.reshape(tf.io.decode_raw(
        parsed_features['probe_r'], tf.uint16), [2, 64, 64])

    fea = tf.transpose(fea, [2, 1, 0])
    lab_k = tf.transpose(lab_k, [2, 1, 0])
    lab_r = tf.transpose(lab_r, [2, 1, 0])
    prob_r = tf.transpose(prob_r, [2, 1, 0])

    prms, sc_x, sc_yk, sc_yr, sc_pr = tf.split(
        parsed_features['meta'], [4, 9, 2, 2, 2])

    # 3x3 cbed-kernel
    sh_x = tf.TensorShape((64, 64, PRM.X_SHAPE[2]))
    x = cast_float32(fea, sc_x, sh_x)

    # Label and probe
    lab_k = cast_wave_float32(lab_k, sc_yk)
    lab_r = cast_wave_float32(lab_r, sc_yr)
    prob_r = cast_wave_float32(prob_r, sc_pr)


    return x, lab_k, lab_r, prob_r, prms


@tf.function
def fcn_rnd_no_sample(x, yk, yr, probe, p):
    rnd = tf.random.uniform([])
    x = tf.cond(
        tf.greater(p, rnd),
        lambda: tf.tile(probe[..., 0, tf.newaxis] ** 2, [1, 1, 9]),
        lambda: x,
    )
    yk = tf.cond(tf.greater(p, rnd), lambda: probe, lambda: yk)
    yr = tf.cond(tf.greater(p, rnd), lambda: tf_ifft2d_A_p(
        tf_normalise_to_one_amp(yk)), lambda: yr)
    return x, yk, yr


@tf.function
def comp_probe_k(prms: tf.Tensor) -> tf.Tensor:
    pk = tf_probe_function(prms[0], prms[1], prms[2], 64, [-1, 0.001], domain='k', type="complex", refine=False)
    pk = tf_normalise_to_one_complex(pk)
    pk = tf.stack([tf.math.abs(pk), tf.math.angle(pk)], axis=-1)
    return pk


@tf.function
def fcn_decode_train(x, lab_k, lab_r, prob_r, prms):
    """Decoding/Preprocessing of HDF5 Dataset for Training (random Poisson)"""

    prob_k = comp_probe_k(prms)
    lab_k = tf_fft2d_A_p(lab_r)

    # Random no-sample
    # x, lab_k, lab_r = fcn_rnd_no_sample(x, lab_k, lab_r, prob_k, 0.05)

    # Add Poisson Noise for random dose
    rnd = tf.random.uniform(
        [1], minval=0, maxval=len(PRM.dose), dtype=tf.int32)
    d = tf.gather(PRM.dose, rnd)
    x = tf.squeeze(tf.random.poisson([1], x * d))

    if PRM.scale_cbeds:
        x = fcn_weight_cbeds(x, prms[3])

    x = tf.concat([x, prob_r], -1)

    msk_k = tf_binary_mask(prob_k[..., 0])[..., tf.newaxis]
    msk_r = tf_binary_mask_r(prms[0], prms[1], prms[2], 64)[..., tf.newaxis]
    y = tf.concat([lab_k, lab_r, prob_r, msk_r, msk_k], -1)

    return x, y


@tf.function
def fcn_decode_val(x, lab_k, lab_r, prob_r, prms):
    """Decoding/Preprocessing of HDF5 Dataset for Validation (random Poisson, doses same)"""

    prob_k = comp_probe_k(prms)
    lab_k = tf_fft2d_A_p(lab_r)

    # Generate a seed based on sample meta-data
    prm_prod = tf.reduce_prod(prms[0:2])
    s1 = tf.cast(tf.round(prm_prod), tf.int64)
    s2 = tf.cast(tf.round(prm_prod + prms[2] * 100), tf.int64)

    # Apply noise from seed
    rnd = tf.random.stateless_uniform(
        [1], seed=(s1, s2), minval=0, maxval=len(PRM.dose), dtype=tf.int32
    )
    d = tf.gather(PRM.dose, rnd)
    x = tf.squeeze(tf.random.stateless_poisson(
        [1]+x.shape.as_list(), lam=(x * d), seed=(s1, s2), dtype=tf.float32))

    if PRM.scale_cbeds:
        x = fcn_weight_cbeds(x, prms[3])

    x = tf.concat([x, prob_r], -1)

    msk_k = tf_binary_mask(prob_k[..., 0])[..., tf.newaxis]
    msk_r = tf_binary_mask_r(prms[0], prms[1], prms[2], 64)[..., tf.newaxis]
    y = tf.concat([lab_k, lab_r, prob_r, msk_r, msk_k], -1)

    return x, y


@tf.function
def fcn_weight_cbeds(cbeds, step_size):
    """Weighting of input CBEDs, according to step size"""
    d1 = (1 / step_size) / 50
    d2 = (1 / tf.math.sqrt(2 * step_size**2)) / 50
    ind_1 = [[1], [3], [5], [7]]
    ind_2 = [[0], [2], [6], [8]]

    cbeds = tf.transpose(cbeds, [2, 0, 1])
    cbeds = tf.tensor_scatter_nd_update(
        cbeds, ind_1, tf.gather_nd(cbeds, ind_1)*d1)
    cbeds = tf.tensor_scatter_nd_update(
        cbeds, ind_2, tf.gather_nd(cbeds, ind_2)*d2)
    cbeds = tf.transpose(cbeds, [1, 2, 0])

    return cbeds


def datasetPipeline(filepaths, is_training, prms):
    """HDF5 Dataset-Pipeline"""
    datasets = list()
    for path in filepaths:
        datasets.append(tf.data.TFRecordDataset(
            path, num_parallel_reads=tf.data.experimental.AUTOTUNE).map(fcn_decode_tfrecords))
        tf.print(path)

    if is_training:
        dataset = tf.data.experimental.sample_from_datasets(datasets)
        # Add the augmentations to the dataset
        dataset = dataset.map(
            fcn_decode_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        # dataset = dataset.shuffle(4096)
        dataset = dataset.repeat(prms["epochs"])
    else:
        dataset = tf.data.experimental.sample_from_datasets(
            datasets, seed=1310)
        dataset = dataset.map(
            fcn_decode_val, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Set the batchsize
    dataset = dataset.batch(prms["batch_size"], drop_remainder=is_training)
    dataset = dataset.prefetch(8)

    return dataset


def getDatasets(prms):
    if PRM.debug:
        n_train_data = prms["batch_size"] * 10
        validation_steps = 8
    else:
        n_train_data = PRM.n_train
        validation_steps = (
            np.min([PRM.n_val, int(n_train_data * 0.05)]) // prms["batch_size"]
        )

    steps_per_epoch = n_train_data // prms["batch_size"]

    PRM.scale_cbeds = prms["scale_cbeds"]

    training_ds = datasetPipeline(prms["train_path"], True, prms)
    validation_ds = datasetPipeline(prms["val_path"], False, prms)

    return training_ds, validation_ds, validation_steps, steps_per_epoch


def getTestDataset(filename, batch_size):
    """Return Dataset for Visualisation"""
    prms = {
        "train_path": filename,
        "epochs": None,
        "batch_size": batch_size
    }
    dataset = datasetPipeline(filename, False, prms)

    return dataset
