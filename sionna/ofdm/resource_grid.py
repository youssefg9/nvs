#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to the resource grid"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

from .pilot_pattern import PilotPattern, EmptyPilotPattern, KroneckerPilotPattern # pylint: disable=line-too-long
from sionna.utils import flatten_last_dims, flatten_dims, split_dim
import matplotlib.pyplot as plt
from matplotlib import colors


class ResourceGrid():
    # pylint: disable=line-too-long
    r"""Defines a `ResourceGrid` spanning multiple OFDM symbols and subcarriers.

    Parameters
    ----------
        num_ofdm_symbols : int
            Number of OFDM symbols.

        fft_size : int
            FFT size (, i.e., the number of subcarriers).

        subcarrier_spacing : float
            The subcarrier spacing in Hz.

        num_tx : int
            Number of transmitters.

        num_streams_per_tx : int
            Number of streams per transmitter.

        cyclic_prefix_length : int
            Length of the cyclic prefix.

        num_guard_carriers : int
            List of two integers defining the number of guardcarriers at the
            left and right side of the resource grid.

        dc_null : bool
            Indicates if the DC carrier is nulled or not.

        pilot_pattern : One of [None, str, PilotPattern, list]
            The pilot pattern configuration. If a list is provided, multiple
            patterns are merged. Overlapping pilots raise an error.

            - `None` defaults to an empty pilot pattern.
            - A `str` must be one of `"kronecker"` or `"empty"`.
              (`"kronecker"` requires `pilot_ofdm_symbol_indices`.)
            - A :class:`~sionna.ofdm.PilotPattern` object is used directly.
            - A `list` of any combination of the above merges them all.

        pilot_ofdm_symbol_indices : List, int
            List of indices of OFDM symbols reserved for pilot transmissions.
            Only needed if `pilot_pattern="kronecker"` or if the list includes
            `"kronecker"`. Defaults to `None`.

        dtype : tf.Dtype
            Datatype for internal calculations and output.
            Defaults to `tf.complex64`.
    """
    def __init__(self,
                 num_ofdm_symbols,
                 fft_size,
                 subcarrier_spacing,
                 num_tx=1,
                 num_streams_per_tx=1,
                 cyclic_prefix_length=0,
                 num_guard_carriers=(0,0),
                 dc_null=False,
                 pilot_pattern=None,
                 pilot_ofdm_symbol_indices=None,
                 dtype=tf.complex64):
        super().__init__()
        self._dtype = dtype
        self._num_ofdm_symbols = num_ofdm_symbols
        self._fft_size = fft_size
        self._subcarrier_spacing = subcarrier_spacing
        self._cyclic_prefix_length = int(cyclic_prefix_length)
        self._num_tx = num_tx
        self._num_streams_per_tx = num_streams_per_tx
        self._num_guard_carriers = np.array(num_guard_carriers)
        self._dc_null = dc_null
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

        # The setter merges multiple patterns if a list is passed.
        self.pilot_pattern = pilot_pattern

        self._check_settings()

    @property
    def cyclic_prefix_length(self):
        """Length of the cyclic prefix."""
        return self._cyclic_prefix_length

    @property
    def num_tx(self):
        """Number of transmitters."""
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        """Number of streams  per transmitter."""
        return self._num_streams_per_tx

    @property
    def num_ofdm_symbols(self):
        """The number of OFDM symbols of the resource grid."""
        return self._num_ofdm_symbols

    @property
    def num_resource_elements(self):
        """Number of resource elements."""
        return self._fft_size*self._num_ofdm_symbols

    @property
    def num_effective_subcarriers(self):
        """Number of subcarriers used for data and pilot transmissions."""
        n = self._fft_size - self._dc_null - np.sum(self._num_guard_carriers)
        return n

    @property
    def effective_subcarrier_ind(self):
        """Returns the indices of the effective subcarriers."""
        num_gc = self._num_guard_carriers
        sc_ind = range(num_gc[0], self.fft_size-num_gc[1])
        if self.dc_null:
            sc_ind = np.delete(sc_ind, self.dc_ind-num_gc[0])
        return sc_ind

    @property
    def num_data_symbols(self):
        """Number of resource elements used for data transmissions."""
        n = self.num_effective_subcarriers * self._num_ofdm_symbols - \
               self.num_pilot_symbols
        return tf.cast(n, tf.int32)

    @property
    def num_pilot_symbols(self):
        """Number of resource elements used for pilot symbols."""
        return self.pilot_pattern.num_pilot_symbols

    @property
    def num_zero_symbols(self):
        """Number of empty resource elements."""
        n = (self._fft_size-self.num_effective_subcarriers) * \
               self._num_ofdm_symbols
        return tf.cast(n, tf.int32)

    @property
    def num_guard_carriers(self):
        """Number of left and right guard carriers."""
        return self._num_guard_carriers

    @property
    def dc_ind(self):
        """Index of the DC subcarrier.

        If ``fft_size`` is odd, the index is (``fft_size``-1)/2.
        If ``fft_size`` is even, the index is ``fft_size``/2.
        """
        return int(self._fft_size/2 - (self._fft_size%2==1)/2)

    @property
    def fft_size(self):
        """The FFT size."""
        return self._fft_size

    @property
    def subcarrier_spacing(self):
        """The subcarrier spacing [Hz]."""
        return self._subcarrier_spacing

    @property
    def ofdm_symbol_duration(self):
        """Duration of an OFDM symbol with cyclic prefix [s]."""
        return (1. + self.cyclic_prefix_length/self.fft_size) \
                / self.subcarrier_spacing

    @property
    def bandwidth(self):
        """The occupied bandwidth [Hz]: ``fft_size*subcarrier_spacing``."""
        return self.fft_size*self.subcarrier_spacing

    @property
    def num_time_samples(self):
        """The number of time-domain samples occupied by the resource grid."""
        return (self.fft_size + self.cyclic_prefix_length) \
                * self._num_ofdm_symbols

    @property
    def dc_null(self):
        """Indicates if the DC carrier is nulled or not."""
        return self._dc_null

    # ------------------------------------------------
    # Pilot pattern logic (allow single or multiple)
    # ------------------------------------------------
    @property
    def pilot_pattern(self):
        """Returns the used PilotPattern (possibly merged)."""
        return self._pilot_pattern

    @pilot_pattern.setter
    def pilot_pattern(self, value):
        # If None => empty pattern
        if value is None:
            value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)

        # Single PilotPattern object
        if isinstance(value, PilotPattern):
            self._pilot_pattern = value
            return

        # Single string
        if isinstance(value, str):
            assert value in ["kronecker", "empty"], \
                f"Unknown pilot pattern: {value}"
            if value=="empty":
                self._pilot_pattern = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
            elif value=="kronecker":
                # Need pilot_ofdm_symbol_indices
                assert self._pilot_ofdm_symbol_indices is not None,\
                    "Must provide pilot_ofdm_symbol_indices for 'kronecker'."
                self._pilot_pattern = KroneckerPilotPattern(
                                          self,
                                          self._pilot_ofdm_symbol_indices,
                                          dtype=self._dtype)
            return

        # If it's a list => merge them
        if isinstance(value, list):
            self._pilot_pattern = self._merge_pilot_patterns(value)
            return

        raise ValueError("Unsupported pilot_pattern. Must be None, 'empty', "
                         "'kronecker', a PilotPattern, or a list of these.")

    def _merge_pilot_patterns(self, pattern_list):
        """
        Merges multiple pilot patterns (or strings/None) into a single pattern.
        Overlapping pilot resources raise ValueError.
        """
        # Start from empty
        base_pp = EmptyPilotPattern(self._num_tx,
                                    self._num_streams_per_tx,
                                    self._num_ofdm_symbols,
                                    self.num_effective_subcarriers,
                                    dtype=self._dtype)
        combined_mask = tf.identity(base_pp.mask)  # int32
        combined_pilots_4d = tf.zeros_like(tf.cast(combined_mask, self._dtype))

        def to_pilot_pattern(item):
            """Convert item to a PilotPattern (handles None, str, PilotPattern)."""
            if item is None:
                return EmptyPilotPattern(self._num_tx,
                                         self._num_streams_per_tx,
                                         self._num_ofdm_symbols,
                                         self.num_effective_subcarriers,
                                         dtype=self._dtype)
            if isinstance(item, PilotPattern):
                return item
            if isinstance(item, str):
                assert item in ["kronecker", "empty"], \
                    f"Unknown pilot pattern: {item}"
                if item == "empty":
                    return EmptyPilotPattern(self._num_tx,
                                             self._num_streams_per_tx,
                                             self._num_ofdm_symbols,
                                             self.num_effective_subcarriers,
                                             dtype=self._dtype)
                if item == "kronecker":
                    assert self._pilot_ofdm_symbol_indices is not None,\
                        ("Must provide pilot_ofdm_symbol_indices "
                         "for 'kronecker'.")
                    return KroneckerPilotPattern(
                                self,
                                self._pilot_ofdm_symbol_indices,
                                dtype=self._dtype)
            raise ValueError("Items in pilot_pattern list must be None, "
                             "str, or PilotPattern objects.")

        # Merge them one by one
        for item in pattern_list:
            pp = to_pilot_pattern(item)
            # shape check
            if pp.mask.shape != base_pp.mask.shape:
                raise ValueError("PilotPattern shape mismatch when merging.")
            # overlap detection:
            overlap_bool = tf.logical_and(tf.equal(combined_mask, 1),
                                          tf.equal(pp.mask, 1))
            if tf.reduce_any(overlap_bool):
                raise ValueError("Overlapping pilot allocations in pilot_pattern list.")

            # merge mask: OR => final=1 where either is 1
            union_bool = tf.logical_or(tf.equal(combined_mask,1),
                                       tf.equal(pp.mask,1))
            combined_mask = tf.where(union_bool, 1, 0)

            # place pilot symbols from pp into combined_pilots_4d
            for t_ind in range(self._num_tx):
                for s_ind in range(self._num_streams_per_tx):
                    submask = pp.mask[t_ind, s_ind]  # int32
                    coords = tf.where(tf.equal(submask, 1))
                    # -- cast coords to int32 to match tx_col/st_col
                    coords = tf.cast(coords, tf.int32)

                    pilot_vals = pp.pilots[t_ind, s_ind]  # shape [#pilots]
                    num_coords = tf.shape(coords)[0]
                    tx_col = tf.fill([num_coords,1], t_ind)
                    st_col = tf.fill([num_coords,1], s_ind)
                    coords_4d = tf.concat([tx_col, st_col, coords], axis=1)

                    combined_pilots_4d = tf.tensor_scatter_nd_update(
                        combined_pilots_4d,
                        coords_4d,
                        pilot_vals)

        # Now build a final PilotPattern from combined_mask + combined_pilots_4d
        pilot_counts = tf.reduce_sum(tf.cast(tf.equal(combined_mask,1), tf.int32),
                                     axis=(-2, -1))  # shape [num_tx, num_streams]
        minc = tf.reduce_min(pilot_counts)
        maxc = tf.reduce_max(pilot_counts)
        if minc != maxc:
            raise ValueError("Each transmitter/stream must have the same pilot "
                             "count after merging. Sionna's PilotPattern expects "
                             "equal pilot counts for all streams.")

        # Convert 4D pilot array back to shape [tx, stream, #pilots]
        final_pilot_list = []
        for t_ind in range(self._num_tx):
            row_list = []
            for s_ind in range(self._num_streams_per_tx):
                mask_2d = combined_mask[t_ind, s_ind]  # shape [T, F_eff], int32
                coords = tf.where(tf.equal(mask_2d, 1))
                coords = tf.cast(coords, tf.int32)  # ensure int32 again
                pilot_2d = combined_pilots_4d[t_ind, s_ind]
                row_vals = tf.gather_nd(pilot_2d, coords)
                row_list.append(row_vals)
            final_pilot_list.append(tf.stack(row_list, axis=0))
        final_pilots_3d = tf.stack(final_pilot_list, axis=0)

        merged_pp = PilotPattern(mask=combined_mask,
                                 pilots=final_pilots_3d,
                                 trainable=False,
                                 normalize=False,
                                 dtype=self._dtype)
        return merged_pp

    def _check_settings(self):
        """Validate that all properties define a valid resource grid"""
        assert self._num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert self._fft_size > 0, \
            "`fft_size` must be positive`."
        assert self._cyclic_prefix_length>=0, \
            "`cyclic_prefix_length must be nonnegative."
        assert self._cyclic_prefix_length<=self._fft_size, \
            "`cyclic_prefix_length` cannot be longer than `fft_size`."
        assert self._num_tx > 0, \
            "`num_tx` must be positive`."
        assert self._num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert len(self._num_guard_carriers)==2, \
            "`num_guard_carriers` must have two elements."
        assert np.all(np.greater_equal(self._num_guard_carriers, 0)), \
            "`num_guard_carriers` must have nonnegative entries."
        assert np.sum(self._num_guard_carriers)<=self._fft_size-self._dc_null,\
            "Total number of guardcarriers cannot be larger than `fft_size`."
        assert self._dtype in [tf.complex64, tf.complex128], \
            "dtype must be tf.complex64 or tf.complex128"
        return True

    def build_type_grid(self):
        """Returns a tensor indicating the type of each resource element.

        Resource elements can be one of

        - 0 : Data symbol
        - 1 : Pilot symbol
        - 2 : Guard carrier symbol
        - 3 : DC carrier symbol

        Output
        ------
        : [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.int32
            Tensor indicating for each transmitter and stream the type of
            the resource elements of the corresponding resource grid.
            The type can be one of [0,1,2,3] as explained above.
        """
        shape = [self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols]
        gc_l = 2*tf.ones(shape+[self._num_guard_carriers[0]], tf.int32)
        gc_r = 2*tf.ones(shape+[self._num_guard_carriers[1]], tf.int32)
        dc   = 3*tf.ones(shape + [tf.cast(self._dc_null, tf.int32)], tf.int32)

        mask = self.pilot_pattern.mask
        split_ind = self.dc_ind - self._num_guard_carriers[0]

        rg_type = tf.concat(
            [
                gc_l,                 # Left Guards
                mask[...,:split_ind], # Data & pilots
                dc,                   # DC
                mask[...,split_ind:], # Data & pilots
                gc_r                  # Right guards
            ],
            axis=-1
        )
        return tf.cast(rg_type, tf.int32)

    def show(self, tx_ind=0, tx_stream_ind=0):
        """Visualizes the resource grid for a specific transmitter and stream.

        Input
        -----
        tx_ind : int
            Indicates the transmitter index.

        tx_stream_ind : int
            Indicates the index of the stream.

        Output
        ------
        : `matplotlib.figure`
            A handle to a matplot figure object.
        """
        fig = plt.figure()
        data = self.build_type_grid()[tx_ind, tx_stream_ind]
        cmap = colors.ListedColormap([
            [60/256,   8/256,  72/256],   # data
            [45/256,  91/256, 128/256],   # pilot
            [45/256, 172/256, 111/256],   # guard
            [250/256,228/256,  62/256]    # DC
        ])
        bounds=[0,1,2,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(np.transpose(data), interpolation="nearest",
                         origin="lower", cmap=cmap, norm=norm,
                         aspect="auto")
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5,3.5],
                            orientation="vertical", shrink=0.8)
        cbar.set_ticklabels(["Data", "Pilot", "Guard carrier", "DC carrier"])
        plt.title("OFDM Resource Grid")
        plt.ylabel("Subcarrier Index")
        plt.xlabel("OFDM Symbol")
        plt.xticks(range(0, data.shape[0]))
        return fig


class ResourceGridMapper(Layer):
    # pylint: disable=line-too-long
    r"""ResourceGridMapper(resource_grid, dtype=tf.complex64, **kwargs)

    Maps a tensor of modulated data symbols to a ResourceGrid.

    This layer takes as input a tensor of modulated data symbols
    and maps them together with pilot symbols onto an
    OFDM :class:`~sionna.ofdm.ResourceGrid`. The output can be
    converted to a time-domain signal with the
    :class:`~sionna.ofdm.Modulator` or further processed in the
    frequency domain.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_data_symbols], tf.complex
        The modulated data symbols to be mapped onto the resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        The full OFDM resource grid in the frequency domain.
    """
    def __init__(self, resource_grid, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._resource_grid = resource_grid

    def build(self, input_shape): # pylint: disable=unused-argument
        """Precompute a tensor of shape
        [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        which is prefilled with pilots and stores indices
        to scatter data symbols.
        """
        self._rg_type = self._resource_grid.build_type_grid()
        self._pilot_ind = tf.where(self._rg_type==1)
        self._data_ind = tf.where(self._rg_type==0)

    def call(self, inputs):
        # Map pilots on empty resource grid
        pilots = flatten_last_dims(self._resource_grid.pilot_pattern.pilots, 3)
        template = tf.scatter_nd(self._pilot_ind,
                                 pilots,
                                 self._rg_type.shape)
        template = tf.expand_dims(template, -1)

        # Broadcast the resource grid template to batch_size
        batch_size = tf.shape(inputs)[0]
        new_shape = tf.concat([tf.shape(template)[:-1], [batch_size]], 0)
        template = tf.broadcast_to(template, new_shape)

        # Flatten the inputs and put batch_dim last for scatter update
        inputs = tf.transpose(flatten_last_dims(inputs, 3))
        rg = tf.tensor_scatter_nd_update(template, self._data_ind, inputs)
        rg = tf.transpose(rg, [4, 0, 1, 2, 3])

        return rg

class ResourceGridDemapper(Layer):
    # pylint: disable=line-too-long
    r"""ResourceGridDemapper(resource_grid, stream_management, dtype=tf.complex64, **kwargs)

    Extracts data-carrying resource elements from a resource grid.

    This layer takes as input an OFDM :class:`~sionna.ofdm.ResourceGrid` and
    extracts the data-carrying resource elements. In other words, it implements
    the reverse operation of :class:`~sionna.ofdm.ResourceGridMapper`.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    stream_management : StreamManagement
        An instance of :class:`~sionna.mimo.StreamManagement`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    : [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size, data_dim]
        The full OFDM resource grid in the frequency domain.
        The last dimension `data_dim` is optional. If `data_dim`
        is used, it refers to the dimensionality of the data that should be
        demapped to individual streams. An example would be LLRs.

    Output
    ------
    : [batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim]
        The data that were mapped into the resource grid.
        The last dimension `data_dim` is only returned if it was used for the
        input.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._stream_management = stream_management
        self._resource_grid = resource_grid

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[...,:num_data_symbols]

    def call(self, y): # pylint: disable=arguments-renamed

        # y has shape
        # [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols,...
        # ..., fft_size, data_dim]

        # If data_dim is not provided, add a dummy dimension
        if len(y.shape)==5:
            y = tf.expand_dims(y, -1)

        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_rx, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers, data dim]
        y = tf.gather(y, self._resource_grid.effective_subcarrier_ind, axis=-2)

        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, data_dim, batch_size]
        y = tf.transpose(y, [1, 2, 3, 4, 5, 0])

        # Merge num_rx amd num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,...
        #  ...,num_effective_subcarriers, data_dim, batch_size]
        y = flatten_dims(y, 2, 0)

        # Put first dimension into the right ordering
        stream_ind = self._stream_management.stream_ind
        y = tf.gather(y, stream_ind, axis=0)

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compared to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., data_dim, batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        y = split_dim(y, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarriers,...
        #  ..., data_dim, batch_size]
        y = flatten_dims(y, 2, 2)

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols, data_dim, batch_size]
        y = tf.gather(y, self._data_ind, batch_dims=2, axis=2)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams, num_data_symbols]
        y = tf.transpose(y, [4, 0, 1, 2, 3])

        # Squeeze data_dim
        if y.shape[-1]==1:
            y = tf.squeeze(y, -1)

        return y

class RemoveNulledSubcarriers(Layer):
    # pylint: disable=line-too-long
    r"""RemoveNulledSubcarriers(resource_grid, **kwargs)

    Removes nulled guard and/or DC subcarriers from a resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex64
        Full resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex64
        Resource grid without nulled subcarriers.
    """
    def __init__(self, resource_grid, **kwargs):
        self._sc_ind = resource_grid.effective_subcarrier_ind
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.gather(inputs, self._sc_ind, axis=-1)
