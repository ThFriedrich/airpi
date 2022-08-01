from tensorflow.keras.utils import plot_model
import tensorflow as tf
from numpy import sqrt
from math import pi
from numpy.fft import fftshift, ifftshift, fft2, ifft2

# compute datatype
floatx = tf.keras.mixed_precision.global_policy().compute_dtype

# constants for tensorflow
tf_pi = tf.constant(pi, dtype=floatx)

# electron rest mass * 2
em2 = tf.constant(1021.9979, dtype=tf.float32)

# Planck's constant * speed of light
hc = tf.constant(12.3984244, dtype=tf.float32)

# @tf.function
def interp2dcomplex(y_cpx=None, y_amp=None, y_phase=None, out_size=[64,64], output='complex'):
    '''
    Interpolate complex data
    Input:
        y_cpx: complex data
        or
        y_amp: amplitude data
        y_phase: phase data
        out_size: output size
        output: 'complex' or 'real'
    Output:
        y_cpx: interpolated complex data
        or
        y_amp: interpolated amplitude data
        y_phase: interpolated phase data
    '''
    if y_cpx is None:
        assert y_amp is not None and y_phase is not None
        y_cpx = tf.cast(y_amp, tf.complex64) * tf.exp(1j*tf.cast(y_phase, tf.complex64))
    else:
        assert(y_cpx.dtype == tf.complex64)
    y_cpx = y_cpx[tf.newaxis,..., tf.newaxis]
    re = tf.image.resize(tf.math.real(y_cpx), out_size)
    im = tf.image.resize(tf.math.imag(y_cpx), out_size)
    cpx = tf.squeeze(tf.complex(re, im))
    if output=='complex':
        return cpx
    else:
        return tf.math.abs(cpx), tf.math.angle(cpx)

def fft2d(A):
    """2D FFT using numpy"""
    return fftshift(fft2(ifftshift(A)))


def ifft2d(A):
    """2D IFFT using numpy"""
    return fftshift(ifft2(ifftshift(A)))


@tf.function
def tf_fft2d(two_d_array):
    """2D Fourier Transform using tensorflow"""
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(two_d_array)))


@tf.function
def tf_ifft2d(two_d_array):
    """2D Inverse Fourier Transform using tensorflow"""
    return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(two_d_array)))


@tf.function
def tf_fft2d_A_p(A_p: tf.Tensor, complex_out:tf.bool=False) -> tf.Tensor:
    """
    Fourier Transform of a 2d wave function as 3d array A_p, with shape (n_x, n_y, 2)
    A_p[:, :, 0] = amplitude
    A_p[:, :, 1] = phase
    """
    nx = tf.cast(tf.shape(A_p)[0],tf.complex64)
    A_p = tf_cast_complex(A_p[..., 0], A_p[..., 1])
    A_p = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(A_p)))/nx

    if complex_out:
        return A_p
    else:
        return tf.stack([tf.math.abs(A_p), tf.math.angle(A_p)], axis=-1)


@tf.function
def tf_ifft2d_A_p(A_p: tf.Tensor, complex_out:tf.bool=False) -> tf.Tensor:
    """
    Inverse Fourier Transform of a 2d wave function as 3d array A_p, with shape (n_x, n_y, 2)
    A_p[:, :, 0] = amplitude
    A_p[:, :, 1] = phase
    """
    nx = tf.cast(tf.shape(A_p)[0],tf.complex64)
    A_p = tf_cast_complex(A_p[..., 0], A_p[..., 1])
    A_p = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(A_p)))*nx
    if complex_out:
        return A_p
    else:
        return tf.stack([tf.math.abs(A_p), tf.math.angle(A_p)], axis=-1)


@tf.function
def tf_rAng_2_mrad(E0: tf.Tensor, rA: tf.Tensor) -> tf.Tensor:
    """
    Conversion of reciprocal Angstrom to mrad
    in tensorflow
    E0: energy in keV
    rA: reciprocal Angstrom
    """
    la = hc / tf.math.sqrt(E0 * (em2 + E0))
    return rA * la / 1e-3


def rAng_2_mrad(E0: float, rA: float) -> float:
    """
    Conversion of reciprocal Angstrom to mrad
    in plain Python
    E0: energy in keV
    rA: reciprocal Angstrom
    """
    la = hc / sqrt(E0 * (em2 + E0))
    return rA * la / 1e-3


def plot_graph(model, name):
    """Plot graph of model"""
    plot_model(
        model,
        to_file=name,
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )


@tf.function
def tf_pow01(x: tf.Tensor) -> tf.Tensor:
    """
    Raise tensor x to the power of 0.1
    """
    return tf.pow(x, 0.1)


def tf_sin_cos2rad(si, co):
    """
    Convert sine and cosine to radians
    Input:
        si: sine
        co: cosine
    Output:
        rad: radians
    """
    si = tf.minimum(1.0, tf.maximum(-1.0, si))
    theta = tf.math.asin(si)

    q2 = tf.math.logical_and(tf.greater(si, 0.0), tf.less(co, 0.0))
    q4 = tf.math.logical_and(tf.less(si, 0.0), tf.less(co, 0.0))
    qm = tf.math.logical_and(tf.less(co, 0.0), tf.equal(si, 0.0))

    theta = tf.where(q2, tf_pi - theta, theta)
    theta = tf.where(q4, -tf_pi - theta, theta)
    theta = tf.where(qm, theta + tf_pi, theta)

    return theta


@tf.function
def tf_sin_cos_decomposition(phase):
    """
    Decomposition of phase into sine and cosine, using tensorflow
    Input:
        phase: phase
    Output:
        sin: sine
        cos: cosine
    """
    l_sin = tf.math.sin(phase)
    l_cos = tf.math.cos(phase)
    return l_sin, l_cos


@tf.function
def tf_normalise_to_one(wave_int: tf.Tensor) -> tf.Tensor:
    """
    Normalise wave function intensity to 1, using tensorflow
    Input:
        wave_int: wave function intensity
    Output:
        wave_int: normalised wave function intensity
    """
    return wave_int / tf.reduce_sum(wave_int)


@tf.function
def tf_normalise_to_one_amp(wave: tf.Tensor) -> tf.Tensor:
    """
    Normalise wave function amplitude to Intensity = 1, using tensorflow
    Input:
        wave_amp: wave function amplitude
    Output:
        wave_amp: normalised wave function amplitude
    """
    wave_int =  wave[...,0]**2
    wave_amp =  tf.math.sqrt(wave_int / tf.reduce_sum(wave_int))
    return tf.stack([wave_amp, wave[...,1]], axis=-1)

@tf.function
def tf_normalise_to_one_complex(wave: tf.Tensor) -> tf.Tensor:
    """
    Normalise wave function amplitude to Intensity = 1, using tensorflow
    Input:
        wave: wave function complex
    Output:
        wave_amp: normalised wave function complex
    """
    wave_int =  tf.math.abs(wave)**2
    wave_amp =  tf.math.sqrt(wave_int / tf.reduce_sum(wave_int))
    return tf_cast_complex(wave_amp, tf.math.angle(wave))


@tf.function
def tf_mrad_2_rAng(E0: tf.Tensor, th: tf.Tensor) -> tf.Tensor:
    """
    Conversion of mrad to reciprocal Angstrom, using tensorflow
    Input:
        E0: energy in keV
        th: mrad
    Output:
        rA: reciprocal Angstrom
    """
    la = hc / tf.math.sqrt(E0 * (em2 + E0))
    return th * 1e-3 / la


@tf.function
def tf_scherzer_df(E0: tf.Tensor, C3) -> tf.Tensor:
    """
    Compute Scherzer defocus, using tensorflow
    Input:
        E0: energy in keV
        C3: 3rd order spherical aberration (mm)
    Output:
        df: defocus in mm
    """
    la = hc / tf.math.sqrt(E0 * (em2 + E0))
    C1 = (1.5 * (C3 * 1e7) * la) ** 0.5  # unit: angstrom
    return C1


@tf.function
def tf_aberration(
    E0: tf.Tensor, ab: tf.Tensor, x: tf.Tensor, y: tf.Tensor
) -> tf.Tensor:
    """
    Compute aberration function, using tensorflow
    Input:
        E0: energy in keV
        ab: aberration parameters [C1 (A), C3(mm)], if ab[0] = 0, then Scherzer defocus is used
        x: probe space x-coordinates in reciprocal Angstrom
        y: probe space y-coordinates in reciprocal Angstrom
    Output:
        ab_func: aberration function
    """
    la = hc / tf.math.sqrt(E0 * (em2 + E0))
    x = x * 1e-3
    y = y * 1e-3
    if ab[0] == -1:
        c0 = tf_scherzer_df(E0, ab[1])
    else:
        c0 = ab[0]
    r = x**2 + y**2
    # only consider C1 (A) and C3 (mm)
    phase = (
        2
        * tf_pi
        / (la * 1e-10)
        * (-0.5 * (c0 * 1e-10) * r + 0.25 * (ab[1] * 1e-3) * r**2)
    )
    return -1 * phase


@tf.function
def tf_probe_k(
    E0: tf.Tensor,
    aperture: tf.Tensor,
    gmax: tf.Tensor,
    nx: tf.Tensor,
    aberration=[0, 0],
):
    la = hc / tf.math.sqrt(E0 * (em2 + E0))

    px_1 = tf.linspace(-gmax, 0.0, tf.cast(nx / 2.0 + 1, tf.int32))
    px_2 = tf.linspace(px_1[-2], px_1[1], tf.cast(nx / 2.0 - 1, tf.int32))
    # forcing zero at center, change from rA to mrad
    px = tf.concat([px_1, px_2], 0) * la / 1e-3
    y, x = tf.meshgrid(px, px)
    ab = tf_aberration(E0, aberration, x, y)
    r = (x**2.0 + y**2.0) ** 0.5
    mask_k_p = tf.where((r <= aperture), ab, 0.0)
    mask_k_i = tf.where((r <= aperture), 1.0, 0.0)  # only 1 or zero

    return mask_k_i, mask_k_p


@tf.function
def tf_probe_function(
    E0: tf.Tensor,
    aperture: tf.Tensor,
    gmax: tf.Tensor,
    nx: tf.Tensor,
    aberration=[0, 0],
    domain="k",
    type="real",
) -> tf.Tensor:

    mask_k_i, mask_k_p = tf_probe_k(E0, aperture, gmax, nx, aberration)

    if domain == "k":
        if type == "real":
            return mask_k_i
        elif type == "complex":
            return tf.stack([mask_k_i, mask_k_p], -1)

    elif domain == "r":
        mask_k = tf_cast_complex(mask_k_i, mask_k_p)
        mask_r_i, mask_r_p = tf_switch_space(mask_k)

        if type == "real":
            return mask_r_i
        elif type == "complex":
            return tf.stack([mask_r_i, mask_r_p], -1)


@tf.function
def tf_switch_space(wave, space="k"):
    """
    Switch between k-space and r-space, using tensorflow
    Input:
        wave: wave function (complex)
        space: 'k' or 'r'
    Output:
        amp: amplitude
        phase: phase
    """
    # amp = tf.math.abs(wave)
    if space == "r":
        wave_new = tf_fft2d(wave)/64
    elif space == "k":
        wave_new = tf_ifft2d(wave)*64
    # scale = tf.reduce_sum(amp) / tf.reduce_sum(amp_new)
    # scale = tf.reduce_sum(amp,axis=[-1,-2],keepdims=True) / tf.reduce_sum(amp_new,axis=[-1,-2],keepdims=True)
    # amp_new *= scale
    amp_new = tf.math.abs(wave_new)
    phase_new = tf.math.angle(wave_new)

    return amp_new, phase_new


@tf.function
def tf_cast_complex(amp: tf.Tensor, phase: tf.Tensor) -> tf.Tensor:
    """
    Cast amplitude and phase to complex number, using tensorflow
    Input:
        amp: amplitude
        phase: phase
    Output:
        wave: wave function (complex)
    """
    return tf.cast(amp, tf.complex64) * tf.math.exp(tf.cast(phase, tf.complex64) * 1j)

@tf.function
def tf_binary_mask(probe_k:tf.Tensor, threshold:float = 0.5) -> tf.Tensor:
    """
    Generate binary mask from probe_k
    Input:
        probe_k: probe function in k-space (complex)
        threshold: relative threshold for binary mask (0...1)
    Output:
        mask: binary mask
    """
    amp = tf.math.abs(probe_k)

    if len(amp.shape) == 3:
        amp_max = tf.reduce_max(amp, axis=[1, 2], keepdims=True)
    else:
        amp_max = tf.reduce_max(amp)
        
    amp_threshold = amp_max * threshold
    mask = tf.where(amp >= amp_threshold, 1.0, 0.0)
    return mask


@tf.function
def pred_dict(y_tensor, btw_filt=None):
    """
    create a dictionary of predicted results for use in differnt metrics and loss functions, using tensorflow.
    Input:
        y_tensor: tensor of predicted results (NN output)
    Output:
        r: dictionary of predicted results in real space
        k: dictionary of predicted results in k-space
        Each dictionary contains:
            amp: amplitude
            phase: phase
            r: real part
            i: imaginary part
            sin: sinusoidal part of phase
            cos: cosine part of phase
    """
    r = dict()
    k = dict()
    y_tensor = tf.where(tf.math.is_nan(y_tensor), tf.zeros_like(y_tensor), y_tensor)
    k["amp_sc"] = y_tensor[...,0]
    k['phase'] = y_tensor[...,1]
    k["amp"] = k["amp_sc"]**5
    k['wv'] = tf_cast_complex(k["amp"], k['phase'])
    k["int"] = tf.math.pow(k["amp"],2)
    k["sin"], k["cos"] = tf_sin_cos_decomposition(k["phase"])

    r["amp"], r["phase"] = tf_switch_space(k['wv'], space="r")
    r["wv"] = tf_cast_complex(r["amp"], r["phase"])
    r["amp_sc"] = tf.math.pow(r["amp"],0.2)
    r["sin"], r["cos"] = tf_sin_cos_decomposition(r["phase"])
    r["int"] = tf.math.abs(tf.math.pow(r["wv"],2))

    return r, k

# @tf.function
# def pred_dict(y_tensor, btw_filt=None):
#     """
#     create a dictionary of predicted results for use in differnt metrics and loss functions, using tensorflow.
#     Input:
#         y_tensor: tensor of predicted results (NN output)
#     Output:
#         r: dictionary of predicted results in real space
#         k: dictionary of predicted results in k-space
#         Each dictionary contains:
#             amp: amplitude
#             phase: phase
#             r: real part
#             i: imaginary part
#             sin: sinusoidal part of phase
#             cos: cosine part of phase
#     """
#     r = dict()
#     k = dict()
#     y_tensor = tf.where(tf.math.is_nan(y_tensor), tf.zeros_like(y_tensor), y_tensor)

#     k["amp_sc"], k["sin"], k["cos"] = tf.unstack(y_tensor, axis=-1)
#     k["amp"] = tf.math.pow(k["amp_sc"],5)
#     # if btw_filt is not None:
#     #     k["amp"] *= btw_filt
#     k["phase"] = tf_sin_cos2rad(k["sin"], k["cos"])
#     kw = tf_cast_complex(k["amp"], k["phase"])
#     k["int"] = tf.math.pow(tf.math.abs(kw),2)
#     # k["r"] = tf.math.real(kw)
#     # k["i"] = tf.math.imag(kw)

#     r["amp"], r["phase"] = tf_switch_space(kw, space="k")
#     r["sin"], r["cos"] = tf_sin_cos_decomposition(r["phase"])
#     r["wv"] = tf_cast_complex(r["amp"], r["phase"])
#     r["int"] = tf.math.abs(tf.math.pow(r["wv"],2))
#     # r["r"] = tf.math.real(r["wv"])
#     # r["i"] = tf.math.imag(r["wv"])
    
#     r["amp"] = tf.where(tf.math.is_nan(r["amp"]), tf.zeros_like(r["amp"]), r["amp"])
#     r["int"] = tf.where(tf.math.is_nan(r["int"]), tf.zeros_like(r["int"]), r["int"])
#     r["amp_sc"] = tf.math.pow(r["amp"],0.2)

#     return r, k


@tf.function
def true_dict(y_tensor):
    """
    create a dictionary of true results for use in differnt metrics and loss functions, using tensorflow.
    Input:
        y_tensor: tensor of true results
    Output:
        r: dictionary of true results in real space
        k: dictionary of true results in k-space
        Each dictionary contains:
            amp: amplitude
            phase: phase
            r: real part
            i: imaginary part
            sin: sinusoidal part of phase
            cos: cosine part of phase
    """
    r = dict()
    k = dict()

    k["amp"], k["phase"], r["amp"], r["phase"], probe, probe_phase, msk_r, msk_k = tf.unstack(y_tensor, axis=-1)
    k["amp_sc"] = k["amp"] ** 0.2
    kw = tf_cast_complex(k["amp"], k["phase"])
    r["probe"] = tf_cast_complex(probe, probe_phase)
    k["probe"], _ = tf_switch_space(r["probe"], space="r")
    k["int"] = tf.math.abs(kw) ** 2
    k["sin"], k["cos"] = tf_sin_cos_decomposition(k["phase"])
    k['msk'] = tf.cast(msk_k,tf.float32)
    # k["r"] = tf.math.real(kw)
    # k["i"] = tf.math.imag(kw)

    rw = tf_cast_complex(r["amp"], r["phase"])
    r["int"] = tf.math.abs(rw) ** 2
    r["sin"], r["cos"] = tf_sin_cos_decomposition(r["phase"])
    # r["r"] = tf.math.real(rw)
    # r["i"] = tf.math.imag(rw)
    r["obj"] = tf.math.angle(rw/r["probe"])
    r["obj"] = tf.where(tf.math.is_nan(r["obj"]), tf.zeros_like(r["obj"]), r["obj"])
    r["amp_sc"] = r["amp"] ** 0.2
    r['msk'] = tf.cast(msk_r,tf.float32)
    return r, k


@tf.function
def tf_com(images):
    """
    Compute centre of mass of images, using tensorflow
    Input:
        images: tensor of images
    Output:
        com: centre of mass
        offset: offset of centre of mass from centre of image
    """
    nb, nx, ny = images.shape[-3:]
    # Make array of coordinates (each row contains three coordinates)
    jj, kk = tf.meshgrid(tf.range(nx), tf.range(ny), indexing="ij")
    coords = tf.stack([tf.reshape(jj, (-1,)), tf.reshape(kk, (-1,))], axis=-1)
    coords = tf.cast(coords, images.dtype)
    # Rearrange input into one vector per volume
    images_flat = tf.reshape(images, [-1, nx * ny, 1])
    # Compute total mass for each volume
    total_mass = tf.reduce_sum(images_flat, axis=1)
    # Compute centre of mass
    com = tf.reduce_sum(images_flat * coords, axis=1) / total_mass
    offset = (nx / 2, ny / 2) - com

    return com, offset


@tf.function
def pad2d(x: tf.Tensor, pad):
    """
    Adds zero-padding to a 2D or 3D tensor.
    Inputs:
        x: 2D or 3D Tensor to be padded
        pad: Integer, padding to be applied to x.
    Outputs:
        x_padded: Padded tensor.
    """
    dp = [pad, pad]
    if len(tf.shape(x)) == 3:
        dz = [0, 0]
        x = tf.pad(x, [dz, dp, dp])
    else:
        x = tf.pad(x, [dp, dp])

    return x

@tf.function
def tf_bessel_root(E0, conv_angle):
    """
    Compute the 0th root of the Bessel function of the first kind.
    Inputs:
        E0: Energy in keV
        conv_angle: Convergence angle in mrad
    Outputs:
        root: Root of the Bessel function of the first kind
    """
    rA = tf_mrad_2_rAng(E0, conv_angle) * 2
    root = (3.8317 / rA) / tf_pi
    return root

@tf.function
def tf_g_space(E0, gmax, nx, unit="mrad"):

    if unit == "mrad":
        lim = tf_rAng_2_mrad(E0, gmax)
    else:
        lim = gmax

    px_1 = tf.linspace(-lim, 0.0, tf.cast(nx / 2.0 + 1, tf.int32))
    px_2 = tf.linspace(px_1[-2], px_1[1], tf.cast(nx / 2.0 - 1, tf.int32))
    px = tf.concat([px_1, px_2], 0)
    y, x = tf.meshgrid(px, px)

    r = (x**2.0 + y**2.0) ** 0.5
    return r

@tf.function
def tf_binary_mask_r(E0, conv_angle, gmax, nx):
    """
    Compute the binary mask for the real space probe.
    Inputs:
        E0: Energy in keV
        conv_angle: Convergence angle in mrad
        gmax: Maximum radius of the mask in m
        nx: Number of pixels in the mask
    Outputs:
        mask: Binary mask of the real space probe
    """

    root = tf_bessel_root(E0, conv_angle)
    r = tf_g_space(E0, gmax, nx, unit="rAng")
    mask_r = tf.where((r <= root), 1.0, 0.0)
    return mask_r


    mask = tf.math.less_equal(tf.math.abs(tf.math.sqrt(tf.math.abs(rA))), gmax * root)
    return mask

@tf.function
def tf_FourierShift2D(x: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """
    `tf_FourierShift2D(x, delta)`

    Tensorflow implementation of subpixel shifting. Based on the original script (FourierShift2D.m)
    by Tim Hutt and its Python adaptation by Suyog Jadhav:
    https://gist.github.com/IAmSuyogJadhav/6b659413dc821d2fb00f290a189da9c1

    ### Description

    Shifts x by delta cyclically. Uses the fourier shift theorem.
    Real inputs should give real outputs.
    By Tim Hutt, 26/03/2009
    Small fix thanks to Brian Krause, 11/02/2010

    ### Parameters

    `x`: Tensorflow Tensor, required
        The 2D matrix that is to be shifted. Must be complex valued.

    `delta`: List, required
        The amount of shift to be done in x and y directions. The 0th index should be
        the shift in the x direction, and the 1st index should be the shift in the y
        direction.

        e.g., For a shift of +2 in x direction and -3 in y direction,
            delta = [2, -3]

    ### Returns

    `y`: The input matrix `x` shifted by the `delta` amount of shift in the
        corresponding directions.
    """
    # The size of the matrix.
    B = tf.shape(x)[0]
    N = tf.shape(x)[1]
    M = tf.shape(x)[2]

    # FFT of our possibly padded input signal.
    X = tf.signal.fft2d(x)

    # The floors take care of odd-length signals.
    y_arr = tf.tile(
        tf.expand_dims(
            tf.concat(
                [
                    tf.range(tf.floor(N / 2), dtype=tf.int32),
                    tf.range(tf.floor(-N / 2), 0, dtype=tf.int32),
                ],
                0,
            ),
            0,
        ),
        [B, 1],
    )

    x_arr = tf.tile(
        tf.expand_dims(
            tf.concat(
                [
                    tf.range(tf.floor(M / 2), dtype=tf.int32),
                    tf.range(tf.floor(-M / 2), 0, dtype=tf.int32),
                ],
                0,
            ),
            0,
        ),
        [B, 1],
    )
    pi2 = 2 * tf.cast(tf_pi, delta.dtype)
    pi_delta_x = tf.cast(
        pi2
        * tf.broadcast_to(delta[:, 0, tf.newaxis], tf.shape(x_arr))
        * tf.cast(x_arr, delta.dtype)
        / tf.cast(N, delta.dtype),
        tf.complex64,
    )
    pi_delta_y = tf.cast(
        pi2
        * tf.broadcast_to(delta[:, 1, tf.newaxis], tf.shape(y_arr))
        * tf.cast(y_arr, delta.dtype)
        / tf.cast(M, delta.dtype),
        tf.complex64,
    )
    y_shift = tf.exp(-1j * pi_delta_x)
    x_shift = tf.exp(-1j * pi_delta_y)

    # Force conjugate symmetry. Otherwise this frequency component has no
    # corresponding negative frequency to cancel out its imaginary part.
    if tf.math.mod(N, 2) == 0:
        indices = tf.stack([tf.range(B), tf.tile([N // 2], [B])], axis=1)
        update = tf.cast(tf.math.real(x_shift[:, N // 2]), tf.complex64)
        x_shift = tf.tensor_scatter_nd_update(x_shift, indices, update)

    if tf.math.mod(M, 2) == 0:
        indices = tf.stack([tf.range(B), tf.tile([M // 2], [B])], axis=1)
        update = tf.cast(tf.math.real(y_shift[:, M // 2]), tf.complex64)
        y_shift = tf.tensor_scatter_nd_update(y_shift, indices, update)

    y_shift = y_shift[:, None, :]  # Shape = (B, 1, N)
    x_shift = x_shift[:, :, None]  # Shape = (B, M, 1)

    Y = X * tf.broadcast_to((x_shift * y_shift), tf.shape(X))

    # Invert the FFT.
    y = tf.signal.ifft2d(Y)

    return y


# @tf.function
def tf_butterworth_filter2D(im_size, cutoff, order=5, shape="ci"):
    """
    `tf_butterworth_filter2D(im_size, cutoff, order, shape)`
    ### Description

    Tensorflow implementation of a 2D-Butterworth Filter, intended for frequency filtering in fourier space

    ### Parameters

    `im_size`: List, required
        Size of the filter in pixels in x and y, e.g. [256, 256]


    `cutoff`: float, required
        The start of the relative angular position of the cutoff ramp, as fraction.
        e.g.: for a filter keeping signal of ~90% of x- and y use cutoff=0.9

    `order`: amp
        Order of the Butterworth filter. The higher the order, the sharper the transition, defaults to 5

    `shape`: string
        use shape='sq' for a square filter or shape='ci' for a circular filter, defaults to 'ci'

    ### Returns

    `f`: The 2D-filter matrix with values between 0 and 1
    """
    rw = im_size[0]
    co = im_size[1]
    order2 = 2 * order
    cutoff2 = cutoff / 2
    [x, y] = tf.meshgrid(
        (tf.range(0, co) - int(tf.experimental.numpy.fix(co / 2))) / co,
        (tf.range(0, rw) - int(tf.experimental.numpy.fix(rw / 2))) / rw,
    )
    if shape == "sq":  # Squared filter window
        fx = 1 + (x / cutoff2) ** order2
        fy = 1 + (y / cutoff2) ** order2
        f = 1 / (fx * fy)
    else:  # Circular filter window
        r = tf.math.sqrt(x**2 + y**2)
        f = 1 / (1 + (r / cutoff2) ** order2)

    return tf.cast(f, floatx)
