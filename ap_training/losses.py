import tensorflow as tf
from ap_architectures.utils import tf_binary_mask, true_dict, pred_dict, tf_butterworth_filter2D
from ap_utils.util_fcns import PRM
import matplotlib.pyplot as plt

btw_flt = tf_butterworth_filter2D(im_size=[64, 64], cutoff=0.95, order=18, shape="ci")

def loss(prms):
    b_normalize = tf.constant(False, dtype=tf.bool)
    loss_fcns_r = [px_p, px_as, px_pw, px_int, px_obj]
    loss_fcns_k = [px_p, px_as, px_pw, px_int, px_cst_int]
    lo_r = []
    for ix in prms["loss_prms_r"]:
        lo_r.append(tf.constant(ix, dtype=tf.float32))
    lo_k = []
    for ix in prms["loss_prms_k"]:
        lo_k.append(tf.constant(ix, dtype=tf.float32))

    @tf.function
    def loss(y_true, y_pred):
        ytr, ypr, ytk, ypk = xy2dict(
            y_true, y_pred, space_out="b", plot_out=PRM.debug
        )
        ls = tf.constant(0.0, dtype=tf.float32)
        if lo_r[0] > 0.0:
            # ls += ecn(ytr, ypr)
            for ix, fcn in enumerate(loss_fcns_r):
                if lo_r[ix + 1] > 0.0:
                    ls += (
                        tf.cast(fcn(ytr, ypr, b_normalize), tf.float32)
                        * lo_r[ix + 1]
                        * lo_r[0]
                    )
        if lo_k[0] > 0.0:
            # ls += tf.cast(ecn(ytk, ypk), tf.float32) * 100
            for ix, fcn in enumerate(loss_fcns_k):
                if lo_k[ix + 1] > 0.0:
                    ls += (
                        tf.cast(fcn(ytk, ypk, b_normalize), tf.float32)
                        * lo_k[ix + 1]
                        * lo_k[0]
                    )
        # ls += (1.0 - xc_p(ytk, ypk))
        # ls += (1.0 - xc_obj(ytr, ypr))
        return ls

    return loss


def get_reg_metric(model):
    def loss_reg(y_true, y_pred):
        return tf.reduce_sum(model.losses)

    return loss_reg


def metric_fcns(prms, prms_net=None, model=None):
    """Returns metrics for training"""
    b_normalize = tf.constant(False, dtype=tf.bool)
    metrics = []
    metrics.append(loss(prms))
    if (prms_net != None) and (model != None):
        if prms_net["w_regul"] != None or prms_net["a_regul"] != None:
            metrics.append(get_reg_metric(model))
    metric_fcns_r = [px_pw, px_int, px_as, px_p, px_obj, xc_obj]
    metric_fcns_k = [
        # ecn,
        px_p,
        px_as,
        xc_p,
        px_int,
        px_pw,
        px_a,
        px_cst_int,
        df_ratio
    ]
    for fcn in metric_fcns_r:

        def _fcn(y_true, y_pred, fcn=fcn, space="r"):
            yt, yp = xy2dict(y_true, y_pred, space_out=space)
            return fcn(yt, yp, b_normalize)

        _fcn.__name__ = fcn.__name__ + "_" + "r"
        metrics.append(_fcn)
    for fcn in metric_fcns_k:

        def _fcn(y_true, y_pred, fcn=fcn, space="k"):
            yt, yp = xy2dict(y_true, y_pred, space_out=space)
            return fcn(yt, yp, b_normalize)

        _fcn.__name__ = fcn.__name__ + "_" + "k"
        metrics.append(_fcn)

    return metrics


@tf.function
def fcn_pix_loss(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    ls = tf.math.abs(y_true - y_pred)
    if b_normalize:
        ls_sc = ls / tf.math.maximum(tf.math.abs(y_true), 1e-6)
        return tf.math.reduce_mean(ls_sc)
    else:
        return tf.math.reduce_mean(ls)


@tf.function
def fcn_pix_loss_weighted(
    y_true, y_pred, weight, b_normalize=tf.constant(False, dtype=tf.bool)
):
    ls = tf.math.abs(y_true - y_pred) * weight
    if b_normalize:
        ls_sc = ls / tf.math.maximum(tf.math.abs(y_true) * weight, 1e-6)
        return tf.math.reduce_mean(ls_sc)
    else:
        return tf.math.reduce_mean(ls)


@tf.function
def fcn_pix_loss_L2(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    ls = (y_true - y_pred) ** 2
    if b_normalize:
        ls_sc = ls / tf.math.maximum(y_true ** 2, 1e-3)
        return tf.math.reduce_mean(ls_sc)
    else:
        return tf.math.reduce_mean(ls)


@tf.function
def px_as(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    ls = fcn_pix_loss(y_true["amp_sc"], y_pred["amp_sc"], b_normalize)
    return ls


@tf.function
def px_p(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    ls = fcn_pix_loss(y_true["sin"], y_pred["sin"], b_normalize)
    ls += fcn_pix_loss(y_true["cos"], y_pred["cos"], b_normalize)
    # ls += fcn_pix_loss(y_true["phase"], y_pred["phase"], b_normalize)
    return ls / 2.0


@tf.function
def fcn_ssim(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    """Loss for Structure similarity index"""
    im1 = tf.expand_dims(y_true, -1)
    im2 = tf.expand_dims(y_pred, -1)
    ssim = tf.image.ssim(
        im1, im2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
    )
    dssim = (1 - ssim) / 2

    return tf.math.reduce_mean(dssim)

@tf.function
def fcn_pearson_xc(y_true, y_pred):
    """Pearson correlation coefficient"""
    t_mean = tf.math.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    p_mean = tf.math.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    tc = (y_true - t_mean)
    pc = (y_pred - p_mean)
    cov = tf.math.reduce_sum( tc * pc, axis=[1, 2])
    t_var = tf.math.reduce_sum( tc * tc, axis=[1, 2])
    p_var = tf.math.reduce_sum( pc * pc, axis=[1, 2])
    corr = cov / tf.squeeze(tf.math.sqrt(t_var * p_var))
    # xc = (1 - corr) / 2
    return tf.math.reduce_mean(corr)

    
@tf.function
def ssi_i(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    return fcn_ssim(y_true["amp"], y_pred["amp"])


@tf.function
def ssi_p(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    ls = fcn_ssim(y_true["phase"], y_pred["phase"])
    return ls

@tf.function
def xc_a(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    return fcn_pearson_xc(y_true["amp"], y_pred["amp"])

@tf.function
def xc_p(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    return fcn_pearson_xc(y_true["phase"], y_pred["phase"])

@tf.function
def xc_obj(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    return fcn_pearson_xc(y_true["obj"], y_pred["obj"])

@tf.function
def fcn_weight(y_true):
    amp = y_true["amp"]
    w = amp - tf.math.reduce_min(amp)
    w = w / tf.math.reduce_max(w)

    return w

@tf.function
def px_pw(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    ls = fcn_pix_loss_weighted(
        y_true["sin"], y_pred["sin"], y_true["weight"], b_normalize
    )
    ls += fcn_pix_loss_weighted(
        y_true["cos"], y_pred["cos"], y_true["weight"], b_normalize
    )
    # ls += fcn_pix_loss_weighted(
    #     y_true["phase"], y_pred["phase"], y_true["weight"], b_normalize
    # )
    return ls / 2.0

@tf.function
def px_a(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    ls = fcn_pix_loss(y_true["amp"], y_pred["amp"], False)
    return ls


@tf.function
def df_ratio(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    b_msk = tf.cast(tf_binary_mask(y_true["probe"]),tf.bool)
    df_t = tf.math.maximum(
        tf.math.reduce_sum(tf.where(b_msk, y_true["int"], 0), axis=[1, 2]), 1e-7
    )
    bf_t = tf.math.maximum(
        tf.math.reduce_sum(tf.where(~b_msk, y_true["int"], 0), axis=[1, 2]), 1e-7
    )
    df_p = tf.math.maximum(
        tf.math.reduce_sum(tf.where(b_msk, y_pred["int"], 0), axis=[1, 2]), 1e-7
    )
    bf_p = tf.math.maximum(
        tf.math.reduce_sum(tf.where(~b_msk, y_pred["int"], 0), axis=[1, 2]), 1e-7
    )
    ratio_t = df_t / bf_t
    ratio_p = df_p / bf_p
    ratio = tf.math.reduce_mean(tf.math.abs(ratio_p - ratio_t) / ratio_t)
    return ratio


@tf.function
def px_cst_int(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    wt = tf.math.reduce_sum(y_true["int"], axis=[1, 2])
    wp = tf.math.reduce_sum(y_pred["int"], axis=[1, 2])
    ls = tf.math.reduce_mean(tf.math.abs(wt - wp) / wt)
    return ls

@tf.function
def px_int(y_true, y_pred, b_normalize=tf.constant(True, dtype=tf.bool)):
    ls = fcn_pix_loss(y_true["int"], y_pred["int"], False)
    return ls

@tf.function
def px_real(y_true, y_pred, b_normalize=tf.constant(True, dtype=tf.bool)):
    ls = fcn_pix_loss_weighted(y_true["r"], y_pred["r"], y_true["weight"], True)
    return ls

@tf.function
def px_imag(y_true, y_pred, b_normalize=tf.constant(True, dtype=tf.bool)):
    ls = fcn_pix_loss_weighted(y_true["i"], y_pred["i"], y_true["weight"], True)
    return ls
1
@tf.function
def px_obj(y_true, y_pred, b_normalize=tf.constant(True, dtype=tf.bool)):
    ls = fcn_pix_loss_weighted(tf.math.sin(y_true["obj"]), tf.math.sin(y_pred["obj"]), y_true["weight"], False)
    ls += fcn_pix_loss_weighted(tf.math.cos(y_true["obj"]), tf.math.cos(y_pred["obj"]), y_true["weight"], False)
    ls /= 2
    return ls

@tf.function
def ecn(y_true, y_pred, b_normalize=tf.constant(False, dtype=tf.bool)):
    ph_n = (
        tf.norm(tf.stack([y_pred["sin"], y_pred["cos"]]), ord="euclidean", axis=0) - 1.0
    )
    return tf.math.reduce_mean(tf.math.abs(ph_n))

# plt.figure(figsize=(12, 8))

@tf.function
def xy2dict(
    y_true, y_pred, space_out="b", plot_out=False
):

    ytr, ytk = true_dict(y_true)
    ypr, ypk = pred_dict(y_pred, btw_flt)
    ypr["obj"] = tf.math.angle(tf.math.divide_no_nan(ypr["wv"],ytr["probe"]))

    ytk["weight"] = fcn_weight(ytk)
    ytr["weight"] = fcn_weight(ytr)


    # if plot_out:
    #     plt.clf()
    #     plot_yt_yp(ytr, ypr, ytk, ypk)
    #     plt.pause(0.5)
        # plt.savefig("imm2.png")

    if space_out == "r":
        return ytr, ypr
    elif space_out == "k":
        return ytk, ypk
    else:
        return ytr, ypr, ytk, ypk


def plot_yt_yp(ytr, ypr, ytk, ypk):
    # plt.figure(figsize=(20, 15))
    # for n in range(len(ytr['amp'])):
    n = 0
    # plt.clf()
    plt.subplot(5, 3, 1).set_axis_off()
    plt.imshow(ytk["amp_sc"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 2).set_axis_off()
    plt.imshow(ypk["amp_sc"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 3).set_axis_off()
    plt.imshow((ytk["amp_sc"][n, ...] - ypk["amp_sc"][n, ...]))
    plt.colorbar()

    plt.subplot(5, 3, 4).set_axis_off()
    plt.imshow(ytk["phase"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 5).set_axis_off()
    plt.imshow(ypk["phase"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 6).set_axis_off()
    plt.imshow((ytk["phase"][n, ...] - ypk["phase"][n, ...]))
    plt.colorbar()

    plt.subplot(5, 3, 7).set_axis_off()
    plt.imshow(ytr["amp"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 8).set_axis_off()
    plt.imshow(ypr["amp"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 9).set_axis_off()
    plt.imshow((ytr["amp"][n, ...] - ypr["amp"][n, ...]))
    plt.colorbar()

    plt.subplot(5, 3, 10).set_axis_off()
    plt.imshow(ytr["phase"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 11).set_axis_off()
    plt.imshow(ypr["phase"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 12).set_axis_off()
    plt.imshow((ytr["phase"][n, ...] - ypr["phase"][n, ...]))
    plt.colorbar()

    plt.subplot(5, 3, 13).set_axis_off()
    plt.imshow(ytr["obj"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 14).set_axis_off()
    plt.imshow(ypr["obj"][n, ...])
    plt.colorbar()
    plt.subplot(5, 3, 15).set_axis_off()
    plt.imshow((ytr["obj"][n, ...] - ypr["obj"][n, ...]))
    plt.colorbar()
    # plt.pause(1)
    # plt.close()
    # plt.savefig('imm.png')
