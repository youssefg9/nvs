# srs_pilot_pattern.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS pilot pattern for the nr (5G) sub-package of the Sionna library.

This class builds a PilotPattern from one or more SRSConfig objects.
"""

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig

__all__ = ["SRSPilotPattern"]

class SRSPilotPattern(PilotPattern):
    """
    SRSPilotPattern builds a pilot pattern for SRS signals from one or more transmitters.

    Parameters
    ----------
    srs_configs : list of SRSConfig
        A list of SRS configuration objects, one per transmitter.
    dtype : tf.DType, optional
        Datatype for internal calculations and output (default: tf.complex64).
    """
    def __init__(self, srs_configs, dtype=tf.complex64):
        # Convert a single SRSConfig to a list, if needed.
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        else:
            for cfg in srs_configs:
                assert isinstance(cfg, SRSConfig), "All elements must be SRSConfig instances."
        num_tx = len(srs_configs)
        carrier = srs_configs[0].carrier
        num_sc = carrier.n_size_grid * 12
        num_sym = carrier.num_symbols_per_slot
        for i, cfg in enumerate(srs_configs):
            assert cfg.carrier.num_symbols_per_slot == num_sym, f"SRS config {i} has a different num_symbols_per_slot."
            assert (cfg.carrier.n_size_grid * 12) == num_sc, f"SRS config {i} has a different subcarrier dimension."
        max_ports = max(cfg.num_srs_ports for cfg in srs_configs)

        # Allocate a uniform pilot mask of shape [num_tx, max_ports, num_sym, num_sc]
        mask_all = np.zeros((num_tx, max_ports, num_sym, num_sc), dtype=bool)
        pilot_values = [[None] * max_ports for _ in range(num_tx)]

        for tx in range(num_tx):
            cfg = srs_configs[tx]
            # Obtain the SRS mask (expected shape: [num_sc, num_sym]) and transpose to [num_sym, num_sc]
            srs_mask = cfg.srs_mask().T
            # Obtain the SRS grid.
            srs_grid = cfg.srs_grid  # Ideally shape: [num_srs_ports, num_sc, num_sym]
            # Patch: ensure srs_grid is at least 3D and has the proper first dimension.
            srs_grid = np.atleast_3d(srs_grid)
            if srs_grid.shape[0] != cfg.num_srs_ports:
                # If only one port is returned but more were requested, tile along axis 0.
                srs_grid = np.tile(srs_grid, (cfg.num_srs_ports, 1, 1))
            if srs_grid.shape[1] != num_sc or srs_grid.shape[2] != num_sym:
                raise ValueError(f"SRSConfig {tx}: srs_grid shape {srs_grid.shape} does not match expected [{cfg.num_srs_ports}, {num_sc}, {num_sym}].")
            # Transpose srs_grid to [num_sym, num_sc, num_srs_ports]
            srs_grid_t = np.transpose(srs_grid, [2, 1, 0])
            for p in range(cfg.num_srs_ports):
                mask_all[tx, p, :, :] = srs_mask
                grid_flat = srs_grid_t[:, :, p].flatten(order='C')
                mask_flat = srs_mask.flatten(order='C')
                pilot_vals = grid_flat[mask_flat]
                pilot_values[tx][p] = pilot_vals

        max_len = 0
        for tx in range(num_tx):
            for p in range(max_ports):
                vals = pilot_values[tx][p]
                if vals is not None:
                    max_len = max(max_len, len(vals))
        pilots_all = np.zeros((num_tx, max_ports, max_len), dtype=complex)
        for tx in range(num_tx):
            for p in range(max_ports):
                vals = pilot_values[tx][p]
                if vals is not None:
                    pilots_all[tx, p, :len(vals)] = vals

        super().__init__(
            mask_all,
            pilots_all,
            trainable=False,
            normalize=False,
            dtype=dtype
        )
