#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Classes and functions related to MIMO precoding:
- Zero-Forcing (ZF) precoding
- Regularized ZF (MMSE) precoding
- MRT (MF) precoding
"""

import tensorflow as tf
from sionna.utils import matrix_inv
from sionna import PI
import math

def zero_forcing_precoder(x, h, return_precoding_matrix=False):
    # pylint: disable=line-too-long
    r"""Zero-Forcing (ZF) Precoder

    This function implements ZF precoding for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{G}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^K` is the received signal vector,
    :math:`\mathbf{H}\in\mathbb{C}^{K\times M}` is the known channel matrix,
    :math:`\mathbf{G}\in\mathbb{C}^{M\times K}` is the precoding matrix,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the symbol vector to be precoded,
    and :math:`\mathbf{n}\in\mathbb{C}^K` is a noise vector. It is assumed that
    :math:`K\le M`.

    The precoding matrix :math:`\mathbf{G}` is defined as (Eq. 4.37) [BHS2017]_ :

    .. math::

        \mathbf{G} = \mathbf{V}\mathbf{D}

    where

    .. math::

        \mathbf{V} &= \mathbf{H}^{\mathsf{H}}\left(\mathbf{H} \mathbf{H}^{\mathsf{H}}\right)^{-1}\\
        \mathbf{D} &= \mathop{\text{diag}}\left( \lVert \mathbf{v}_{k} \rVert_2^{-1}, k=0,\dots,K-1 \right).

    This ensures that each stream is precoded with a unit-norm vector,
    i.e., :math:`\mathop{\text{tr}}\left(\mathbf{G}\mathbf{G}^{\mathsf{H}}\right)=K`.
    The function returns the precoded vector :math:`\mathbf{G}\mathbf{x}`.

    Input
    -----
    x : [...,K], tf.complex
        1+D tensor containing the symbol vectors to be precoded.

    h : [...,K,M], tf.complex
        2+D tensor containing the channel matrices

    return_precoding_matrices : bool
        Indicates if the precoding matrices should be returned or not.
        Defaults to False.

    Output
    -------
    x_precoded : [...,M], tf.complex
        Tensor of the same shape and dtype as ``x`` apart from the last
        dimensions that has changed from `K` to `M`. It contains the
        precoded symbol vectors.

    g : [...,M,K], tf.complex
        2+D tensor containing the precoding matrices. It is only returned
        if ``return_precoding_matrices=True``.

    Note
    ----
    If you want to use this function in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    # Compute pseudo inverse for precoding
    g = tf.matmul(h, h, adjoint_b=True)
    g = tf.matmul(h, matrix_inv(g), adjoint_a=True)

    # Normalize each column to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(g)**2, axis=-2, keepdims=True))
    g = g / tf.cast(norm, g.dtype)

    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)
    
    # Precode
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, g
    return x_precoded

#############################################
# 1) Regularized ZF (MMSE) Precoder
#############################################
def regularized_zf_precoder(x, 
                            h, 
                            no,  # noise variance
                            return_precoding_matrix=False):
    # pylint: disable=line-too-long
    r"""
    Regularized Zero-Forcing (MMSE) Precoder

    This function implements the MMSE (also known as regularized-ZF) precoding.
    We consider the same MIMO model:
    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{G}\mathbf{x} + \mathbf{n}

    The precoding matrix :math:`\mathbf{G}` is defined as:
    .. math::
        \mathbf{G} = \mathbf{H}^{\mathsf{H}}
        \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \alpha \mathbf{I}\right)^{-1}
        \mathbf{D},
    where :math:`\alpha` is a regularization term that depends on the noise
    variance, and :math:`\mathbf{D}` ensures column normalization, similar
    to ZF.

    Input
    -----
    x : [..., K], tf.complex
        1+D tensor containing the symbol vectors to be precoded.

    h : [..., K, M], tf.complex
        2+D tensor containing the channel matrices.
        `K` is number of data streams/users, `M` is number of transmit antennas.

    no : float or tf.float32
        Noise variance to be used for regularization.

    return_precoding_matrix : bool
        If `True`, returns the full precoding matrix as well.

    Output
    ------
    x_precoded : [..., M], tf.complex
        The precoded symbol vector(s) of shape [...,M].

    g : [..., M, K], tf.complex
        The regularized ZF precoding matrices. Returned if
        `return_precoding_matrix=True`.
    """
    # H * H^H => shape [..., K, K]
    hh = tf.matmul(h, h, adjoint_b=True)
    # add alpha * I
    k = tf.shape(h)[-2]  # number of data streams = K
    alphaI = tf.eye(k, dtype=h.dtype) * tf.cast(no, h.dtype)
    hh_reg = hh + alphaI

    # invert
    inv_hh_reg = matrix_inv(hh_reg)  # shape [..., K, K]

    # G = H^H (H H^H + alpha I)^(-1)
    g = tf.matmul(h, inv_hh_reg, adjoint_a=True)

    # Column-wise normalization to ensure unit-norm columns
    norm = tf.sqrt(tf.reduce_sum(tf.abs(g)**2, axis=-2, keepdims=True))
    g = g / tf.cast(norm, g.dtype)

    # Precode x
    x_precoded = tf.expand_dims(x, -1)  # [..., K, 1]
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)  # [..., M]

    if return_precoding_matrix:
        return x_precoded, g
    return x_precoded

#############################################
# 2) MRT (Matched Filter) Precoder
#############################################
def mrt_precoder(x,
                 h,
                 return_precoding_matrix=False):
    # pylint: disable=line-too-long
    r"""
    Matched Filter (MRT) Precoder

    This function implements the MRT precoding matrix:
    .. math::
        \mathbf{G} = \mathbf{H}^{\mathsf{H}} \mathbf{D},

    where :math:`\mathbf{D}` ensures each column (per stream) has unit norm.

    Input
    -----
    x : [..., K], tf.complex
        1+D tensor containing the symbol vectors to be precoded.

    h : [..., K, M], tf.complex
        2+D tensor containing the channel matrices.
        `K` is number of data streams, `M` is number of TX antennas.

    return_precoding_matrix : bool
        If `True`, returns the full precoding matrix as well.

    Output
    ------
    x_precoded : [..., M], tf.complex
        The precoded symbol vector(s).

    g : [..., M, K], tf.complex
        The MRT precoding matrices. Returned if `return_precoding_matrix=True`.
    """
    # G = H^H
    g = tf.transpose(h, perm=[0,2,1]) if len(h.shape)>2 else tf.linalg.adjoint(h)

    # Normalize columns to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(g)**2, axis=-2, keepdims=True))
    g = g / tf.cast(norm, g.dtype)

    x_precoded = tf.expand_dims(x, -1)  # [..., K, 1]
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)  # [..., M]

    if return_precoding_matrix:
        return x_precoded, g
    return x_precoded

#############################################
# 3) C2PO Precoder (1-bit iterative)
#############################################
def c2po_precoder(x,
                  h,
                  iterations=5,
                  tau=1e-3,
                  rho=1.0,
                  return_precoding_matrix=False):
    # pylint: disable=line-too-long
    r"""
    C2PO (1-bit Iterative) Precoder

    Implements a simplified 1-bit iterative beamforming approach that tries
    to find a phase-only quantized solution. This is a demonstration of how
    one might incorporate iterative finite-resolution precoding into a
    Sionna-like function, following the references such as [JSTSP2018]_.

    The idea:
    1. Initialize x = H^H s
    2. Use an approximate inverse A ≈ I - τ H^H (I - s s^H / ||s||^2 ) H
    3. Iteratively update x = ρ * A x
    4. Project to 1-bit phase

    The final x is then scaled s.t. columns are unit norm if multi-stream.

    Input
    -----
    x : [..., K], tf.complex
        The symbol vector(s) to be precoded.

    h : [..., K, M], tf.complex
        The channel matrix. K = #streams, M = #antennas.

    iterations : int
        Number of C2PO iterations.

    tau : float
        Step size for approximate inverse.

    rho : float
        "Push factor" scaling after each iteration.

    return_precoding_matrix : bool
        If `True`, returns the approximate precoding matrix as well
        (final 1-bit solution).

    Output
    ------
    x_precoded : [..., M], tf.complex
        The precoded symbol vectors.

    g : [..., M, K], tf.complex
        Approximate final precoding matrix, if requested.
    """

    # We assume single-stream for simplicity in this reference approach.
    # For multi-stream, you'd refine the approach to handle separate columns.
    # Expand dims for consistent matrix ops
    x_expand = tf.expand_dims(x, -1)  # [..., K, 1]

    # H shape: [..., K, M]
    # Initial guess: x0 = H^H s
    h_dagger = tf.transpose(h, [0,2,1]) if len(h.shape)>2 else tf.linalg.adjoint(h)
    x0 = tf.matmul(h_dagger, x_expand) # [..., M, 1]
    # Let's define s_norm^2
    s_power = tf.reduce_sum(tf.abs(x_expand)**2, axis=-2, keepdims=True) # [...,1,1]

    # Approx. inverse: A = I - tau * H^H (I - s s^H / ||s||^2 ) H
    # shape analysis: H^H has shape [..., M, K], (I - s s^H/||s||^2) is [..., K, K], then H is [...,K,M]
    # For single stream, (I - s s^H/||s||^2) might be 0 if K=1. Let's assume K=1 => we skip that. 
    # We'll do the general approach for K>1 if needed:
    #   i.e. A = I_M - tau * H^H (I_K - s s^H / ||s||^2 ) H
    # shape of (I_K - s s^H / ||s||^2): we must build an identity matrix of shape [K,K].
    batch_shape = tf.shape(h)[:-2]
    K = tf.shape(h)[-2]
    M = tf.shape(h)[-1]

    eye_k = tf.eye(K, dtype=h.dtype)
    s_sh = tf.matmul(x_expand, x_expand, adjoint_b=True) / (s_power+1e-9)
    middle = eye_k - s_sh  # [...,K,K]

    # H^H middle H => shape [..., M, M]
    hm = tf.matmul(h_dagger, tf.matmul(middle, h))
    eye_m = tf.eye(M, dtype=h.dtype)
    A = eye_m - tf.cast(tau, h.dtype)*hm  # [..., M, M]

    # Iterative updates
    xi = x0  # start
    for _ in range(iterations):
        # xi = rho * A xi
        xi = rho * tf.matmul(A, xi)
        # 1-bit projection: keep only phase
        # amplitude = 1/sqrt(M) or some scaling
        phase = tf.math.angle(xi)
        # amplitude can be 1 / sqrt(M)
        amp = 1.0 / tf.sqrt(tf.cast(M, xi.dtype))
        xi = amp * tf.exp(1j*phase)

    x_precoded = tf.squeeze(xi, axis=-1) # [..., M]

    if return_precoding_matrix:
        # We treat x_precoded as single-stream precoder if K=1
        # or broadcast to [M,K] identity approach for multi-stream
        # for demonstration, create a matrix that places x_precoded as a column
        x_precoded_2d = tf.expand_dims(x_precoded, -1) # [..., M, 1]
        return x_precoded, x_precoded_2d
    return x_precoded

#############################################
# 4) C3PO Precoder (3-bit iterative)
#############################################
def c3po_precoder(x,
                  h,
                  iterations=5,
                  tau=1e-3,
                  rho=1.0,
                  return_precoding_matrix=False):
    # pylint: disable=line-too-long
    r"""
    C3PO (3-bit Iterative) Precoder

    Similar to C2PO, but with a 3-bit (8-level) phase or amplitude/phase
    quantization. Here, we demonstrate a simple phase-only 3-bit approach
    (i.e., quantizing phase to multiples of :math:`\frac{2\pi}{8}=\frac{\pi}{4}`).

    1. Initialize x = H^H s
    2. Approx inverse: A = I - tau H^H (I - s s^H / ||s||^2 ) H
    3. Iteration: x = rho * A x
    4. 3-bit phase projection

    Input
    -----
    x : [..., K], tf.complex
        Symbol vector.

    h : [..., K, M], tf.complex
        Channel matrix.

    iterations : int
        Number of iterative refinements.

    tau : float
        Step size.

    rho : float
        Push factor for the iterative update.

    return_precoding_matrix : bool
        If True, also returns the final precoding matrix.

    Output
    ------
    x_precoded : [..., M], tf.complex
        3-bit quantized precoded vector.

    g : [..., M, K], tf.complex
        Optional final precoding matrix.
    """
    # Similar to c2po_precoder but do 3-bit phase.
    # 1) Init
    x_expand = tf.expand_dims(x, -1)
    h_dagger = tf.transpose(h, [0,2,1]) if len(h.shape)>2 else tf.linalg.adjoint(h)
    x0 = tf.matmul(h_dagger, x_expand)
    s_power = tf.reduce_sum(tf.abs(x_expand)**2, axis=-2, keepdims=True)

    # 2) Build approximate inverse A
    K = tf.shape(h)[-2]
    M = tf.shape(h)[-1]
    eye_k = tf.eye(K, dtype=h.dtype)
    s_sh = tf.matmul(x_expand, x_expand, adjoint_b=True) / (s_power+1e-9)
    middle = eye_k - s_sh
    hm = tf.matmul(h_dagger, tf.matmul(middle, h))
    eye_m = tf.eye(M, dtype=h.dtype)
    A = eye_m - tf.cast(tau, h.dtype)*hm

    # 3) Iteration
    xi = x0
    for _ in range(iterations):
        xi = rho * tf.matmul(A, xi)
        # 3-bit phase quant: phases to multiples of pi/4
        phase = tf.math.angle(xi)
        phase_q = tf.round(phase / (PI/4.0)) * (PI/4.0)
        # amplitude => 1 / sqrt(M) for demonstration
        amp = 1.0 / tf.sqrt(tf.cast(M, xi.dtype))
        xi = amp * tf.exp(1j * phase_q)

    # Squeeze
    x_precoded = tf.squeeze(xi, axis=-1)

    if return_precoding_matrix:
        x_precoded_2d = tf.expand_dims(x_precoded, -1)
        return x_precoded, x_precoded_2d
    return x_precoded

def grid_of_beams_dft_ula(num_ant,
                          oversmpl=1):
    # pylint: disable=line-too-long
    r""" Computes the Discrete Fourier Transform (DFT) Grid of Beam (GoB)
    coefficients for a uniform linear array (ULA)
    
    The coefficient applied to antenna :math:`n` for beam :math:`m` is expressed
    as:
    
    .. math:: 
        c_n^m = e^{\frac{2\pi n m}{N O}}, \quad n=0,\dots,N-1, \ m=0,\dots,NO
    
    where :math:`N` is the number of antennas ``num_ant`` and :math:`O` is the oversampling
    factor ``oversmpl``. 

    Note that the main lobe of beam :math:`m` points in the azimuth direction 
    :math:`\theta = \mathrm{arc sin} \left( 2\frac{m}{N} \right)` if :math:`m\le
    N/2` and :math:`\theta = \mathrm{arc sin} \left( 2\frac{m-N}{N} \right)` if
    :math:`m\ge N/2`, where :math:`\theta=0` defines the perpendicular to the
    antenna array. 

    Input
    ------
    num_ant : int
        Number of antennas

    oversmpl : int
        Oversampling factor

    Output
    -------
    gob : [num_ant x oversmpl, num_ant], tf.complex
        The :math:`m`-th row contains the `num_ant` antenna coefficients for
        the :math:`m`-th DFT beam
    """
    oversmpl = int(oversmpl)

    # Beam indices: [0, .., num_ant * oversmpl - 1]
    beam_ind = tf.range(num_ant * oversmpl, dtype=tf.float32)[:, tf.newaxis]

    # Antenna indices: [0, .., num_ant - 1]
    antenna_ind = tf.range(num_ant, dtype=tf.float32)[tf.newaxis, :]

    # Combine real and imaginary part and normalize power to 1
    phases = 2 * PI * beam_ind * antenna_ind / (num_ant * oversmpl)
    gob = tf.complex(tf.cos(phases), tf.sin(phases)) / math.sqrt(num_ant)
    return gob

def grid_of_beams_dft(num_ant_v,
                      num_ant_h,
                      oversmpl_v=1,
                      oversmpl_h=1):
    # pylint: disable=line-too-long
    r""" Computes the Discrete Fourier Transform (DFT) Grid of Beam (GoB)
    coefficients for a uniform rectangular array (URA)

    GoB indices are arranged over a 2D grid indexed by :math:`(m_v,m_h)`. 
    The coefficient of the beam with index :math:`(m_v,m_h)` applied to the
    antenna located at row :math:`n_v` and column :math:`n_h` of the rectangular
    array is expressed as:
    
    .. math:: 
        c_{n_v,n_h}^{m_v,m_h} = e^{\frac{2\pi n_h m_v}{N_h O_h}} e^{\frac{2\pi n_h m_h}{N_v O_v}}
    
    where :math:`n_v=0,\dots,N_v-1`, :math:`n_h=0,\dots,N_h-1`,
    :math:`m_v=0,\dots,N_v O_v`, :math:`m_h=0,\dots,N_h O_h`, :math:`N` is the
    number of antennas ``num_ant`` and :math:`O_v,O_h` are the oversampling
    factor ``oversmpl_v``, ``oversmpl_h`` in the vertical and
    horizontal direction, respectively. 

    We can rewrite more concisely the matrix coefficients
    :math:`c^{m_v,m_h}` as follows:

    .. math::
        c^{m_v,m_h} = c^{m_v} \otimes c^{m_h}

    where :math:`\otimes` denotes the Kronecker product and
    :math:`c^{m_v},c^{m_h}` are the ULA DFT beams computed as in
    :func:`~sionna.mimo.grid_of_beams_dft_ula` .

    Such a DFT GoB is, e.g., defined in Section 5.2.2.2.1 [3GPP38214]_.

    Input
    ------
    num_ant_v : int
        Number of antenna rows (i.e., in vertical direction) of the rectangular
        array

    num_ant_h : int
        Number of antenna columns (i.e., in horizontal direction) of the
        rectangular array.

    oversmpl_v : int
        Oversampling factor in vertical direction

    oversmpl_h : int
        Oversampling factor in horizontal direction

    Output
    -------
    gob : [num_ant_v x oversmpl_v, num_ant_h x oversmpl_h, num_ant_v x num_ant_h], tf.complex 
        The elements :math:`[m_v,m_h,:]` contain the antenna coefficients of the
        DFT beam with index pair :math:`(m_v,m_h)`.
    """

    # Compute the DFT coefficients to be applied in the vertical direction
    gob_v = grid_of_beams_dft_ula(num_ant_v, oversmpl=oversmpl_v)
    gob_v = gob_v[:, tf.newaxis, :, tf.newaxis]

    # Compute the DFT coefficients to be applied in the horizontal direction
    gob_h = grid_of_beams_dft_ula(num_ant_h, oversmpl=oversmpl_h)
    gob_h = gob_h[tf.newaxis, :, tf.newaxis, :]

    # Kronecker product:
    # [num_ant_v * oversmpl_v , num_ant_h * oversmpl_v, num_ant_v, num_ant_h]
    coef_vh = tf.math.multiply(gob_h, gob_v)
    # Flatten the last two dimensions to produce 1-dimensional precoding vectors
    # [num_ant_v * oversmpl_v , num_ant_h * oversmpl_v, num_ant_v x num_ant_h]
    coef_vh = flatten_precoding_mat(coef_vh)
    return coef_vh

def flatten_precoding_mat(precoding_mat, by_column=True):
    # pylint: disable=line-too-long
    r"""Flattens a [..., num_ant_v, num_ant_h] precoding matrix associated with
    a rectangular array by producing a [..., num_ant_v x num_ant_h] precoding vector.

    Input
    ------
    precoding_mat : [..., num_antennas_vertical, num_antennas_horizontal], tf.complex 
        Precoding matrix. The element :math:`(i,j)` contains the precoding
        coefficient of the antenna element located at row :math:`i` and column
        :math:`j` of a rectangular antenna array.

    by_column : bool
        If `True`, then flattening occurs on a per-column basis, i.e., the first
        column is appended to the second, and so on. Else, flattening is performed on
        a per-row basis.

    Output
    -------
    : [..., num_antennas_vertical x num_antennas_horizontal], tf.complex 
        Flattened precoding vector
    """

    # Transpose the last two dimensions
    if by_column:
        precoding_mat = tf.linalg.matrix_transpose(precoding_mat)
    # Flatten the last two dimensions
    precoding_vec = tf.reshape(
        precoding_mat, precoding_mat.shape[:-2] + [math.prod(precoding_mat.shape[2:])])
    return precoding_vec

def normalize_precoding_power(precoding_vec, dtype=None, tx_power_list=None):
    # pylint: disable=line-too-long
    r""" Normalizes the beam coefficient power to 1 by default, or to
    ``tx_power_list`` if provided as input.

    Input
    ------
    precoding_vec : [N,M], tf.complex
        Each row contains a set of antenna coefficients whose power is to be normalized.

    dtype : dtype
        dtype of the output. Defaults to None.

    tx_power_list : [N], float
        The :math:`i`-th element defines the power of the :math:`i`-th precoding vector.

    Output
    -------
     : [N,M] tf.complex
        Normalized antenna coefficients.
    """
    if dtype is None:
        dtype = precoding_vec.dtype

    if len(precoding_vec.shape)==1:
        precoding_vec = precoding_vec[tf.newaxis, :]

    if tx_power_list is None:
        # By default, power is normalized to 1
        tx_power_list = [1] * precoding_vec.shape[0]

    precoding_vec_norm = tf.cast(tf.norm(precoding_vec, axis=1), dtype)[
        :, tf.newaxis]
    tx_power = tf.constant(tx_power_list, dtype=dtype)[:, tf.newaxis]

    # Normalize the power of each row
    precoding_vec = tf.math.multiply(tf.math.divide(
        precoding_vec, precoding_vec_norm), tx_power)

    return precoding_vec
