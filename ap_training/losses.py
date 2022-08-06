import tensorflow as tf
from ap_utils.functions import true_dict, pred_dict, tf_butterworth_filter2D
from ap_utils.globals import PRM
import matplotlib.pyplot as plt

btw_flt = tf_butterworth_filter2D((64,64),0.9,12)

@tf.function(jit_compile=True)
def loss(y_true, y_pred):
    ytr, ypr, ytk, ypk = xy2dict(
        y_true, y_pred, space_out="b", plot_out=PRM.debug
    )
    w = tf.reduce_sum(tf.cast(ytk["msk"],tf.uint16),axis=[1,2], keepdims=True)/4096
    l_ak = ik(ytk, ypk, (1-w)) + ask(ytk, ypk, w)
    l_ar = ir(ytr, ypr, (w)) + asr(ytr, ypr)
    l_pk = pk(ytk, ypk, (1-w))
    l_pr = pr(ytr, ypr, (w))
    l_ob = obj(ytr, ypr)
    l_xc_p = (1 - xc_p(ytk, ypk))/2.0
    l_xc_a = (1 - xc_a(ytk, ypk))/2.0
    l_xc_obj = (1 - xc_obj(ytr, ypr))/2.0
    return l_ak + l_ar + l_pk + l_pr + l_xc_p + l_xc_a + l_xc_obj + l_ob

def get_reg_metric(model):
    def loss_reg(y_true, y_pred):
        return tf.reduce_sum(model.losses)

    return loss_reg

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

@tf.function(jit_compile=True)
def MSE(y_true, y_pred, w=None):
    ls = (y_true - y_pred) ** 2
    if w != None:
        ls *= w
    return ls

@tf.function(jit_compile=True)
def MAE(y_true, y_pred, w=None):
    ls = tf.math.abs(y_true - y_pred)
    if w != None:
        ls *= w
    return ls

@tf.function(jit_compile=True)
def RMSE(y_true, y_pred, w=None):
    ls = tf.math.sqrt((y_true - y_pred) ** 2)
    if w != None:
        ls *= w
    return ls

@tf.function(jit_compile=True)
def rE(y_true, y_pred, msk=None, w=None):
    ls = tf.math.divide_no_nan(tf.math.abs(y_true - y_pred),(0.5*tf.math.abs(y_true)))
    ls = tf.math.minimum(ls, 1.0)
    if w != None:
        ls *= w
    return ls

@tf.function(jit_compile=True)
def sMAPE(y_true, y_pred,  w=None):
    d =  (y_true - y_pred)
    if w != None:
        d *= w
    dn = tf.maximum(y_true, MAE(y_true, y_pred))
    ls = tf.math.abs(tf.math.divide_no_nan(d, dn))
    return ls

@tf.function(jit_compile=True)
def sMAPE2(y_true, y_pred,  w=None):
    d =  (y_true - y_pred)
    if w != None:
        d *= w
    mae = MAE(y_true, y_pred)
    dn = tf.maximum(y_true, mae)
    ls = tf.math.abs(tf.math.divide_no_nan(d, dn))
    ls *= tf.math.exp(mae)
    return ls

@tf.function(jit_compile=True)
def cauchy(y_true, y_pred, w=None):
    ls = tf.math.log(1+tf.math.divide_no_nan((y_pred-y_true)**2,y_true**2))
    if w != None:
        ls *= w
    return ls

@tf.function(jit_compile=True)
def ask(y_true, y_pred, w=None):
    ls = MAE(y_true['amp_sc'], y_pred['amp_sc'])
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)

@tf.function(jit_compile=True)
def asr(y_true, y_pred, w=None):
    ls = MAE(y_true['amp_sc'], y_pred['amp_sc'], btw_flt)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)

@tf.function(jit_compile=True)
def ik(y_true, y_pred, w=None):
    ls = MAE(y_true['int'], y_pred['int'])
    ls = tf.math.reduce_sum(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)

@tf.function(jit_compile=True)
def ir(y_true, y_pred, w=None):
    ls = MAE(y_true['int'], y_pred['int'])
    ls = tf.math.reduce_sum(ls, axis=[1,2],keepdims=True)
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)

@tf.function(jit_compile=True)
def pk(y_true, y_pred, w=None):
    ls = (MSE(y_true['sin'], y_pred['sin']) + \
          MSE(y_true['cos'], y_pred['cos']))
    # ls += sMAPE2(y_true['phase'], y_pred['phase'])
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)


@tf.function(jit_compile=True)
def pr(y_true, y_pred, w=None):
    ls = (MSE(y_true['sin'], y_pred['sin'], y_true['weight']) + \
          MSE(y_true['cos'], y_pred['cos'], y_true['weight']))
    # ls += sMAPE2(y_true['phase'], y_pred['phase'], y_true['weight'])
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)


@tf.function(jit_compile=True)
def ar(y_true, y_pred, w=None):
    ls = MAE(y_true['amp'], y_pred['amp'])
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)


@tf.function(jit_compile=True)
def obj(y_true, y_pred, w=None):
    ls = sMAPE(y_true["obj"], y_pred["obj"], w=y_true['weight'])
    if w != None:
        ls *= w
    return tf.math.reduce_mean(ls)


@tf.function(jit_compile=True)
def fcn_pearson_xc(y_true, y_pred):
    """Pearson correlation coefficient"""
    t_mean = tf.math.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    p_mean = tf.math.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    tc = (y_true - t_mean)
    pc = (y_pred - p_mean)
    cov = tf.math.reduce_sum(tc * pc, axis=[1, 2])
    t_var = tf.math.reduce_sum(tc * tc, axis=[1, 2])
    p_var = tf.math.reduce_sum(pc * pc, axis=[1, 2])
    corr = cov / tf.squeeze(tf.math.sqrt(t_var * p_var))
    # xc = (1 - corr) / 2
    return tf.math.reduce_mean(corr)


@tf.function(jit_compile=True)
def xc_a(y_true, y_pred):
    return fcn_pearson_xc(y_true["amp_sc"], y_pred["amp_sc"])


@tf.function(jit_compile=True)
def xc_p(y_true, y_pred):
    return fcn_pearson_xc(y_true["phase"], y_pred["phase"])


@tf.function(jit_compile=False)
def xc_obj(y_true, y_pred):
    return fcn_pearson_xc(y_true["obj"], y_pred["obj"])

@tf.function(jit_compile=True)
def fcn_weight(y_true):
    amp = y_true["amp"]
    w = amp - tf.math.reduce_min(amp, axis=[-1, -2], keepdims=True)
    w = w / tf.math.reduce_max(w, axis=[-1, -2], keepdims=True)
    return w

@tf.function(jit_compile=True)
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

