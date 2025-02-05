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
        # Convert a single SRSConfig to a list.
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
        
        # Allocate a uniform pilot mask: shape [num_tx, max_ports, num_sym, num_sc]
        mask_all = np.zeros((num_tx, max_ports, num_sym, num_sc), dtype=bool)
        # Create a nested list to hold pilot sequences for each transmitter/port.
        pilot_values = [[None] * max_ports for _ in range(num_tx)]
        
        for tx in range(num_tx):
            cfg = srs_configs[tx]
            # Get SRS mask: expected shape (num_sc, num_sym); transpose to (num_sym, num_sc).
            srs_mask = np.asarray(cfg.srs_mask()).T
            # Get SRS grid; expected shape: (num_srs_ports, num_sc, num_sym)
            grid = np.asarray(cfg.srs_grid)
            # --- Force grid to be 3D ---
            if grid.ndim == 2:
                # Assume shape (num_sc, num_sym) and add a new axis.
                grid = grid[np.newaxis, :, :]
            # Now, force first axis to have length equal to cfg.num_srs_ports.
            if grid.shape[0] != cfg.num_srs_ports:
                # Use np.resize which repeats the data as needed.
                grid = np.resize(grid, (cfg.num_srs_ports, num_sc, num_sym))
            if grid.shape != (cfg.num_srs_ports, num_sc, num_sym):
                raise ValueError(f"SRSConfig {tx}: grid shape {grid.shape} does not match expected ({cfg.num_srs_ports}, {num_sc}, {num_sym}).")
            # Transpose grid to shape (num_sym, num_sc, num_srs_ports)
            grid_t = np.transpose(grid, [2, 1, 0])
            # For each port available in this configuration...
            for p in range(cfg.num_srs_ports):
                mask_all[tx, p, :, :] = srs_mask
                # grid_t[:, :, p] has shape (num_sym, num_sc)
                flat_grid = grid_t[:, :, p].flatten(order='C')
                flat_mask = srs_mask.flatten(order='C')
                pilot_seq = flat_grid[flat_mask]
                pilot_values[tx][p] = pilot_seq
            # For ports that are not present (if cfg.num_srs_ports < max_ports), fill with zeros.
            for p in range(cfg.num_srs_ports, max_ports):
                pilot_values[tx][p] = np.zeros(0, dtype=complex)
        
        # Determine the maximum pilot sequence length across all tx and ports.
        global_max_len = 0
        for tx in range(num_tx):
            for p in range(max_ports):
                if pilot_values[tx][p] is not None:
                    global_max_len = max(global_max_len, len(pilot_values[tx][p]))
        # Build a uniform pilots array of shape [num_tx, max_ports, global_max_len].
        pilots_all = np.zeros((num_tx, max_ports, global_max_len), dtype=complex)
        for tx in range(num_tx):
            for p in range(max_ports):
                seq = pilot_values[tx][p]
                if seq is not None:
                    pilots_all[tx, p, :len(seq)] = seq

        super().__init__(
            mask_all,
            pilots_all,
            trainable=False,
            normalize=False,
            dtype=dtype
        )
