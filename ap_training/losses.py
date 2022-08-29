import tensorflow as tf
from ap_utils.functions import true_dict, pred_dict
from ap_utils.globals import PRM
import matplotlib.pyplot as plt


@tf.function
def loss(y_true, y_pred):
    ytr, ypr, ytk, ypk = xy2dict(
        y_true, y_pred, space_out="b", plot_out=PRM.debug
    )
    l_ak = ik(ytk, ypk) + ask(ytk, ypk)
    l_ar = ir(ytr, ypr) + asr(ytr, ypr)
    l_pk = pk(ytk, ypk) * 3.0
    l_pr = pr(ytr, ypr)
    l_ob = obj(ytr, ypr)
    l_xc_obj = (1 - xc_obj(ytr, ypr))/2.0

    return l_ar + l_pr + l_ob + l_xc_obj + l_pk + l_ak

def metrics():
    """Returns metrics for training"""
    metrics = []
    metrics.append(loss)
    metric_fcns_r = [ir, pr, obj, xc_obj]
    metric_fcns_k = [ik, pk, xc_a, xc_p]
    for fcn in metric_fcns_r:
        def _fcn(y_true, y_pred, fcn=fcn, space="r"):
            yt, yp = xy2dict(y_true, y_pred, space_out=space)
            return fcn(yt, yp)
        _fcn.__name__ = fcn.__name__ + "_r"
        metrics.append(_fcn)
    for fcn in metric_fcns_k:
        def _fcn(y_true, y_pred, fcn=fcn, space="k"):
            yt, yp = xy2dict(y_true, y_pred, space_out=space)
            return fcn(yt, yp)
        _fcn.__name__ = fcn.__name__ + "_k"
        metrics.append(_fcn)
       
    return metrics

@tf.function
def MSE(y_true, y_pred, w=None):
    ls = (y_true - y_pred) ** 2
    if w != None:
        ls *= w
    return ls

@tf.function
def ask(y_true, y_pred, w=None):
    ls = MSE(y_true['amp_sc'], y_pred['amp_sc'])
    ls = tf.math.reduce_mean(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)

@tf.function
def asr(y_true, y_pred, w=None):
    ls = MSE(y_true['amp_sc'], y_pred['amp_sc'])
    ls = tf.math.reduce_mean(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)


@tf.function
def ik(y_true, y_pred, w=None):
    ls = MSE(y_true['int'], y_pred['int'])
    ls = tf.math.reduce_sum(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)

@tf.function
def ir(y_true, y_pred, w=None):
    ls = MSE(y_true['int'], y_pred['int'])
    ls = tf.math.reduce_sum(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)

@tf.function
def pk(y_true, y_pred, w=None):
    ls = (MSE(y_true['sin'], y_pred['sin']) + \
          MSE(y_true['cos'], y_pred['cos']))
    ls = tf.math.reduce_mean(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)


@tf.function
def pr(y_true, y_pred, w=None):
    ls = (MSE(y_true['sin'], y_pred['sin']) + \
          MSE(y_true['cos'], y_pred['cos']))
    ls = tf.math.reduce_mean(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)


@tf.function
def obj(y_true, y_pred, w=None):
    ls = MSE(y_true["obj"], y_pred["obj"])
    ls = tf.math.reduce_mean(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)

@tf.function
def fcn_pearson_xc(y_true, y_pred):
    """Pearson correlation coefficient"""
    t_mean = tf.math.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    p_mean = tf.math.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    tc = (y_true - t_mean)
    pc = (y_pred - p_mean)
    cov = tf.math.reduce_sum(tc * pc, axis=[1, 2])
    t_var = tf.math.reduce_sum(tc * tc, axis=[1, 2])
    p_var = tf.math.reduce_sum(pc * pc, axis=[1, 2])
    xc = cov / tf.squeeze(tf.math.sqrt(t_var * p_var))
    return xc


@tf.function
def xc_a(y_true, y_pred, w=None):
    xc = fcn_pearson_xc(y_true["amp_sc"], y_pred["amp_sc"])
    if w != None:
        xc *= w
    return tf.math.reduce_mean(xc)


@tf.function
def xc_p(y_true, y_pred, w=None):
    xc = fcn_pearson_xc(y_true["phase"], y_pred["phase"])
    if w != None:   
        xc *= w
    return tf.math.reduce_mean(xc)


@tf.function
def xc_obj(y_true, y_pred, w=None):
    xc = fcn_pearson_xc(y_true["phase"], y_pred["phase"])
    if w != None:   
        xc *= w
    return tf.math.reduce_mean(xc)

@tf.function
def fcn_weight(y_true):
    amp = y_true["amp_sc"]
    w = amp - tf.math.reduce_min(amp, axis=[-1, -2], keepdims=True)
    w = w / tf.math.reduce_max(w, axis=[-1, -2], keepdims=True)
    return w

@tf.function
def xy2dict(
    y_true, y_pred, space_out="b", plot_out=False):

    ytr, ytk = true_dict(y_true)
    ypr, ypk = pred_dict(y_pred)

    ypr["obj"] = tf.math.angle(tf.math.divide_no_nan(ypr["wv"], ytr["probe"]))

    ytk["weight"] = fcn_weight(ytk)
    ytr["weight"] = fcn_weight(ytr)

    # if plot_out:
    #     plt.clf()
    #     plot_yt_yp(ytr, ypr, ytk, ypk)
    #     plt.pause(0.5)

    if space_out == "r":
        return ytr, ypr
    elif space_out == "k":
        return ytk, ypk
    else:
        return ytr, ypr, ytk, ypk

def plot_yt_yp(ytr, ypr, ytk, ypk):
    n = 0
    # for n in range(ytr["amp_sc"].shape[0]):
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

