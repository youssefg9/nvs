# srs_pilot_pattern.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS pilot pattern for the nr (5G) sub-package of the Sionna library.

supports different numbers of SRS ports across multiple transmitters.
"""

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig

__all__ = ["SRSPilotPattern"]

class SRSPilotPattern(PilotPattern):
    r"""
    OFDM PilotPattern for SRS signals from one or more transmitters, each possibly
    with a different number of SRS ports.

    Parameters
    ----------
    srs_configs : list of SRSConfig
        A list of SRS configuration objects, one per transmitter.
    
    dtype : tf.DType
        Datatype for internal calculations and final output. Defaults to tf.complex64.

    Notes
    -----
    If one transmitter has (for example) 2 SRS ports and another has 1,
    the final arrays are built with a uniform port dimension equal to the maximum
    number of ports among all configurations. Unused port entries are zero-filled.
    """
    def __init__(self, srs_configs, dtype=tf.complex64):
        # Convert a single SRSConfig to a list.
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        else:
            for cfg in srs_configs:
                assert isinstance(cfg, SRSConfig), \
                    "All elements must be SRSConfig instances."

        num_tx = len(srs_configs)
        carrier = srs_configs[0].carrier
        num_sc = carrier.n_size_grid * 12
        num_sym = carrier.num_symbols_per_slot

        # Ensure common dimensions across configurations.
        for i, cfg in enumerate(srs_configs):
            assert cfg.carrier.num_symbols_per_slot == num_sym, \
                f"SRS config {i} has a different num_symbols_per_slot."
            assert (cfg.carrier.n_size_grid * 12) == num_sc, \
                f"SRS config {i} has a different subcarrier dimension."

        # Determine the maximum number of SRS ports among all configs.
        max_ports = max(cfg.num_srs_ports for cfg in srs_configs)

        # Allocate a final mask array of shape [num_tx, max_ports, num_sym, num_sc].
        mask_all = np.zeros([num_tx, max_ports, num_sym, num_sc], dtype=bool)
        # Prepare a nested list to collect pilot symbols per transmitter and port.
        pilot_values = [[None] * max_ports for _ in range(num_tx)]

        # For each transmitter, fill in the mask and pilot values.
        for tx in range(num_tx):
            cfg = srs_configs[tx]
            # Get the SRS mask from the configuration (expected shape: [num_sc, num_sym]).
            srs_mask = cfg.srs_mask().T  # Transpose to shape [num_sym, num_sc].
            # Get the SRS grid from the configuration.
            # Expected original shape: [num_srs_ports, num_sc, num_sym].
            # Reorder to shape: [num_sym, num_sc, num_srs_ports].
            srs_grid = cfg.srs_grid
            srs_grid = np.transpose(srs_grid, [2, 1, 0])
            # For each port in this transmitter’s configuration:
            for p in range(cfg.num_srs_ports):
                mask_all[tx, p, :, :] = srs_mask
                grid_flat = srs_grid[:, :, p].flatten(order='C')
                mask_flat = srs_mask.flatten(order='C')
                pilot_vals = grid_flat[mask_flat]
                pilot_values[tx][p] = pilot_vals
            # Ports beyond cfg.num_srs_ports up to max_ports remain zero.

        # Determine the maximum number of pilot REs among all ports.
        max_len = 0
        for tx in range(num_tx):
            for p in range(max_ports):
                vals = pilot_values[tx][p]
                if vals is not None:
                    max_len = max(max_len, len(vals))
        
        # Build a uniform pilots array of shape [num_tx, max_ports, max_len],
        # zero-padding shorter arrays.
        pilots_all = np.zeros([num_tx, max_ports, max_len], dtype=complex)
        for tx in range(num_tx):
            for p in range(max_ports):
                vals = pilot_values[tx][p]
                if vals is not None:
                    pilots_all[tx, p, :len(vals)] = vals

        # Call the base PilotPattern constructor.
        super().__init__(
            mask_all,          # shape: [num_tx, max_ports, num_sym, num_sc]
            pilots_all,        # shape: [num_tx, max_ports, max_len]
            trainable=False,
            normalize=False,
            dtype=dtype
        )
