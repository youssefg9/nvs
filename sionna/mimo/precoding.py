#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definitions and functions related to OFDM transmit precoding.

Extended to include:
- MMSE (Regularized ZF) Precoding
- MRT Precoding
- C2PO Precoding
- C3PO Precoding
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.mimo import zero_forcing_precoder,regularized_zf_precoder,mrt_precoder,c2po_precoder,c3po_precoder
from sionna.ofdm import RemoveNulledSubcarriers


class ZFPrecoder(Layer):
    # pylint: disable=line-too-long
    r"""ZFPrecoder(resource_grid, stream_management, return_effective_channel=False, dtype=tf.complex64, **kwargs)

    Zero-forcing precoding for multi-antenna transmissions.

    This layer precodes a tensor containing OFDM resource grids using
    the :meth:`~sionna.mimo.zero_forcing_precoder`. For every
    transmitter, the channels to all intended receivers are gathered
    into a channel matrix, based on the which the precoding matrix
    is computed and the input tensor is precoded. The layer also outputs
    optionally the effective channel after precoding for each stream.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    stream_management : StreamManagement
        An instance of :class:`~sionna.mimo.StreamManagement`.

    return_effective_channel : bool
        Indicates if the effective channel after precoding should be returned.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (x, h) :
        Tuple:

    x : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Tensor containing the resource grid to be precoded.

    h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm, fft_size], tf.complex
        Tensor containing the channel knowledge based on which the precoding
        is computed.

    Output
    ------
    x_precoded : [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        The precoded resource grids.

    h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm, num_effective_subcarriers], tf.complex
        Only returned if ``return_effective_channel=True``.
        The effectice channels for all streams after precoding. Can be used to
        simulate perfect channel state information (CSI) at the receivers.
        Nulled subcarriers are automatically removed to be compliant with the
        behavior of a channel estimator.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 return_effective_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def _compute_effective_channel(self, h, g):
        """Compute effective channel after precoding"""

        # Input dimensions:
        # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,...
        #     ..., num_ofdm, fft_size]
        # g: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,
        #     ..., num_streams_per_tx]

        # Transpose h to shape:
        # [batch_size, num_rx, num_tx, num_ofdm, fft_size, num_rx_ant,...
        #  ..., num_tx_ant]
        h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
        h = tf.cast(h, g.dtype)

        # Add one dummy dimension to g to be broadcastable to h:
        # [batch_size, 1, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,...
        #  ..., num_streams_per_tx]
        g = tf.expand_dims(g, 1)

        # Compute post precoding channel:
        # [batch_size, num_rx, num_tx, num_ofdm, fft_size, num_rx_ant,...
        #  ..., num_streams_per_tx]
        h_eff = tf.matmul(h, g)

        # Permute dimensions to common format of channel tensors:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm, fft_size]
        h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])

        # Remove nulled subcarriers:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm, num_effective_subcarriers]
        h_eff = self._remove_nulled_scs(h_eff)

        return h_eff

    def call(self, inputs):

        x, h = inputs
        # x has shape
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #
        # h has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm,...
        # ..., fft_size]

        ###
        ### Transformations to bring h and x in the desired shapes
        ###

        # Transpose x:
        #[batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)

        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])

        # Gather desired channel for precoding:
        # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc_desired = tf.gather(h_pc, self._stream_management.precoding_ind,
                                 axis=1, batch_dims=1)

        # Flatten dims 2,3:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size,...
        #  ..., num_streams_per_tx, num_tx_ant]
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self._dtype)

        ###
        ### ZF precoding
        ###
        x_precoded, g = zero_forcing_precoder(x_precoded,
                                              h_pc_desired,
                                              return_precoding_matrix=True)

        # Transpose output to desired shape:
        #[batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])

        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return (x_precoded, h_eff)
        else:
            return x_precoded


class MMSEPrecoder(Layer):
    # pylint: disable=line-too-long
    r"""MMSEPrecoder(resource_grid, stream_management, noise_variance=1e-3, return_effective_channel=False, dtype=tf.complex64, **kwargs)

    Regularized ZF (MMSE) precoding for multi-antenna transmissions.

    Similar to :class:`ZFPrecoder` but uses :meth:`~sionna.mimo.regularized_zf_precoder`.

    Parameters
    ----------
    resource_grid : ResourceGrid
        The OFDM resource grid configuration.

    stream_management : StreamManagement
        Manages which user streams go to which transmit antennas.

    noise_variance : float
        Noise variance for regularization.

    return_effective_channel : bool
        If `True`, returns the effective channel after precoding.

    dtype : tf.DType
        Datatype for internal calculations and output.

    Input
    -----
    (x, h) :
        Same as :class:`ZFPrecoder`.

    Output
    ------
    x_precoded : [batch, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        The precoded OFDM grids.

    h_eff : [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm, num_effective_subcarriers], tf.complex
        (Optional) Effective channel after precoding.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 noise_variance=1e-3,
                 return_effective_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._noise_variance = noise_variance
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def _compute_effective_channel(self, h, g):
        """Compute effective channel after precoding"""
        # Same logic as ZFPrecoder
        h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
        h = tf.cast(h, g.dtype)
        g = tf.expand_dims(g, 1)
        h_eff = tf.matmul(h, g)
        h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])
        h_eff = self._remove_nulled_scs(h_eff)
        return h_eff

    def call(self, inputs):
        x, h = inputs
        # 1) Reorder x
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)

        # 2) Reorder h
        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])
        h_pc_desired = tf.gather(h_pc,
                                 self._stream_management.precoding_ind,
                                 axis=1,
                                 batch_dims=1)
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self._dtype)

        # 3) MMSE precoding
        x_precoded, g = regularized_zf_precoder(
            x_precoded,
            h_pc_desired,
            no=self._noise_variance,
            return_precoding_matrix=True
        )

        # 4) Reorder output
        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])

        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return x_precoded, h_eff
        return x_precoded


class MRTPrecoder(Layer):
    # pylint: disable=line-too-long
    r"""MRTPrecoder(resource_grid, stream_management, return_effective_channel=False, dtype=tf.complex64, **kwargs)

    MRT (Matched Filter) precoding for multi-antenna transmissions.

    Similar to :class:`ZFPrecoder` but uses :meth:`~sionna.mimo.mrt_precoder`.

    Parameters
    ----------
    resource_grid : ResourceGrid
        OFDM resource grid configuration

    stream_management : StreamManagement
        Manages which user streams go to which transmit antennas

    return_effective_channel : bool
        If `True`, returns the effective channel after precoding

    dtype : tf.DType
        Datatype for internal calculations and output

    Input
    -----
    (x, h) :
        Same shapes as :class:`ZFPrecoder`

    Output
    ------
    x_precoded : [batch, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        The MRT-precoded OFDM grids.

    h_eff : [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm, num_effective_subcarriers], tf.complex
        (Optional) Effective channel.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 return_effective_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def _compute_effective_channel(self, h, g):
        h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
        h = tf.cast(h, g.dtype)
        g = tf.expand_dims(g, 1)
        h_eff = tf.matmul(h, g)
        h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])
        h_eff = self._remove_nulled_scs(h_eff)
        return h_eff

    def call(self, inputs):
        x, h = inputs
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)

        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])
        h_pc_desired = tf.gather(h_pc,
                                 self._stream_management.precoding_ind,
                                 axis=1,
                                 batch_dims=1)
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self._dtype)

        x_precoded, g = mrt_precoder(x_precoded,
                                     h_pc_desired,
                                     return_precoding_matrix=True)

        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])
        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return x_precoded, h_eff
        return x_precoded


class C2POPrecoder(Layer):
    # pylint: disable=line-too-long
    r"""C2POPrecoder(resource_grid, stream_management, iterations=5, tau=1e-3, rho=1.0, return_effective_channel=False, dtype=tf.complex64, **kwargs)

    1-bit iterative precoding for multi-antenna transmissions using C2PO.

    Similar to :class:`ZFPrecoder` but uses :meth:`~sionna.mimo.c2po_precoder`,
    which employs an iterative phase-projection algorithm.  
    This often applies to **low-resolution** digital beamforming.

    Parameters
    ----------
    resource_grid : ResourceGrid
        OFDM resource grid configuration

    stream_management : StreamManagement
        Manages which user streams map to which antennas

    iterations : int
        Number of C2PO iterations

    tau : float
        Step size for approximate inverse

    rho : float
        "Push factor" scaling after each iteration

    return_effective_channel : bool
        If `True`, returns the effective channel after precoding

    dtype : tf.DType
        Datatype for internal calculations and output

    Input
    -----
    (x, h) :
        Same shapes as :class:`ZFPrecoder`

    Output
    ------
    x_precoded : [batch, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        The C2PO-precoded OFDM grids.

    h_eff : [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm, num_effective_subcarriers], tf.complex
        (Optional) Effective channel.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 iterations=5,
                 tau=1e-3,
                 rho=1.0,
                 return_effective_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._iterations = iterations
        self._tau = tau
        self._rho = rho
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def _compute_effective_channel(self, h, g):
        h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
        h = tf.cast(h, g.dtype)
        g = tf.expand_dims(g, 1)
        h_eff = tf.matmul(h, g)
        h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])
        h_eff = self._remove_nulled_scs(h_eff)
        return h_eff

    def call(self, inputs):
        x, h = inputs
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)

        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])
        h_pc_desired = tf.gather(h_pc,
                                 self._stream_management.precoding_ind,
                                 axis=1,
                                 batch_dims=1)
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self._dtype)

        x_precoded, g = c2po_precoder(
            x_precoded,
            h_pc_desired,
            iterations=self._iterations,
            tau=self._tau,
            rho=self._rho,
            return_precoding_matrix=True
        )

        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])
        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return x_precoded, h_eff
        return x_precoded


class C3POPrecoder(Layer):
    # pylint: disable=line-too-long
    r"""C3POPrecoder(resource_grid, stream_management, iterations=5, tau=1e-3, rho=1.0, return_effective_channel=False, dtype=tf.complex64, **kwargs)

    3-bit iterative precoding for multi-antenna transmissions using C3PO.

    Similar to :class:`C2POPrecoder` but uses :meth:`~sionna.mimo.c3po_precoder`,
    which quantizes phase to 3-bit resolution (8 levels). This can also be
    extended to amplitude-phase quantization if desired.

    Parameters
    ----------
    resource_grid : ResourceGrid
        OFDM resource grid

    stream_management : StreamManagement
        Mapping of user streams

    iterations : int
        Number of iterative updates

    tau : float
        Step size for approximate inverse

    rho : float
        "Push factor" after each iteration

    return_effective_channel : bool
        If `True`, returns the effective channel after precoding

    dtype : tf.DType
        Datatype for internal ops

    Input
    -----
    (x, h) :
        Identical shapes as :class:`ZFPrecoder`

    Output
    ------
    x_precoded : [batch, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        The 3-bit quantized precoded grids.

    h_eff : [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm, num_effective_subcarriers], tf.complex
        (Optional) Effective channel.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 iterations=5,
                 tau=1e-3,
                 rho=1.0,
                 return_effective_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._iterations = iterations
        self._tau = tau
        self._rho = rho
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def _compute_effective_channel(self, h, g):
        h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
        h = tf.cast(h, g.dtype)
        g = tf.expand_dims(g, 1)
        h_eff = tf.matmul(h, g)
        h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])
        h_eff = self._remove_nulled_scs(h_eff)
        return h_eff

    def call(self, inputs):
        x, h = inputs
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)

        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])
        h_pc_desired = tf.gather(h_pc,
                                 self._stream_management.precoding_ind,
                                 axis=1,
                                 batch_dims=1)
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self._dtype)

        x_precoded, g = c3po_precoder(
            x_precoded,
            h_pc_desired,
            iterations=self._iterations,
            tau=self._tau,
            rho=self._rho,
            return_precoding_matrix=True
        )

        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])
        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return x_precoded, h_eff
        return x_precoded
