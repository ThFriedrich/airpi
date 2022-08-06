import tensorflow as tf
import os
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import ix_, zeros

from ap_utils.functions import pred_dict, tf_pow01, true_dict
from ap_utils.colormaps import parula
from ap_utils.globals import PRM

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
