# srs_pilot_pattern.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS pilot pattern for the nr (5G) sub-package of the Sionna library.

Revised version that supports different numbers of SRS ports across
multiple transmitters.
"""

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig

__all__ = ["SRSPilotPattern"]

class SRSPilotPattern(PilotPattern):
    r"""
    OFDM PilotPattern for SRS signals from one or more transmitters, each with
    a possibly different number of SRS ports.

    Parameters
    ----------
    srs_configs : list of SRSConfig
        A list of SRS configuration objects, one per transmitter.
    
    dtype : tf.DType
        Datatype for internal calculations and final output. Defaults to `tf.complex64`.

    Notes
    -----
    If one transmitter has (for example) 2 SRS ports, and another has 1,
    we store both in arrays of shape `[num_tx, max_srs_ports, num_symbols, num_sc]`
    for the mask, and `[num_tx, max_srs_ports, num_pilots]` for the pilots.
    Ports beyond the actual number used by a transmitter are left as zero.
    """
    def __init__(self, srs_configs, dtype=tf.complex64):

        # Convert single config to list
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        else:
            for cfg in srs_configs:
                assert isinstance(cfg, SRSConfig), \
                    "All elements must be SRSConfig instances."

        num_tx = len(srs_configs)

        # Basic checks on common carrier dimension
        carrier = srs_configs[0].carrier
        num_sc = carrier.n_size_grid * 12
        num_sym = carrier.num_symbols_per_slot

        for i, cfg in enumerate(srs_configs):
            assert cfg.carrier.num_symbols_per_slot == num_sym, \
                f"SRS config {i} has a different num_symbols_per_slot."
            assert (cfg.carrier.n_size_grid * 12) == num_sc, \
                f"SRS config {i} has a different subcarrier dimension."

        # Find maximum number of ports among all configs
        max_ports = max(cfg.num_srs_ports for cfg in srs_configs)

        # Prepare final mask/pilots arrays
        # mask shape:   [num_tx, max_ports, num_sym, num_sc]
        # pilots shape: [num_tx, max_ports, ?]  -> we determine ? below
        mask_all = np.zeros([num_tx, max_ports, num_sym, num_sc], dtype=bool)

        # We'll store pilots in a two-level structure, then unify later
        # pilot_values[tx][port] = 1D array of pilot symbols
        pilot_values = [[None]*max_ports for _ in range(num_tx)]

        # Fill the mask and pilot_values for each transmitter + port
        for tx in range(num_tx):
            cfg = srs_configs[tx]
            # get srs_mask => shape (subcarriers, symbols)
            # we'll transpose to (symbols, subcarriers)
            srs_mask = cfg.srs_mask().T  # => [num_sym, num_sc]

            # get srs_grid => shape (num_ports, num_sc, num_sym) from SRSConfig
            # we want to reorder to [num_sym, num_sc, num_ports]
            srs_grid = cfg.srs_grid
            # srs_grid: [P, SC, SYM]
            srs_grid = np.transpose(srs_grid, [2,1,0])  # => [SYM, SC, P]

            # For p in this config's actual ports
            for p in range(cfg.num_srs_ports):
                # Mark the mask portion
                mask_all[tx, p, :, :] = srs_mask

                # Flatten the relevant SRS grid to pick out pilot REs
                grid_flat = srs_grid[:,:,p].flatten(order='C')  # shape => [SYM*SC]
                mask_flat = srs_mask.flatten(order='C')
                # pilot symbols are the subset
                pilot_vals = grid_flat[mask_flat]  
                pilot_values[tx][p] = pilot_vals
        
        # Now we need a uniform 3D array for pilots: [num_tx, max_ports, num_pilots]
        # The number of pilot REs can differ among SRS configs or ports, so let's
        # find the maximum length, then zero-pad shorter ones.
        max_len = 0
        for tx in range(num_tx):
            for p in range(max_ports):
                vals = pilot_values[tx][p]
                if vals is not None:
                    max_len = max(max_len, len(vals))
        
        # Build final pilots array
        pilots_all = np.zeros([num_tx, max_ports, max_len], dtype=complex)
        for tx in range(num_tx):
            for p in range(max_ports):
                vals = pilot_values[tx][p]
                if vals is not None:
                    pilots_all[tx, p, :len(vals)] = vals

        # Finally, call the base PilotPattern constructor
        super().__init__(
            mask_all,          # shape: [num_tx, max_ports, num_sym, num_sc]
            pilots_all,        # shape: [num_tx, max_ports, max_len]
            trainable=False,
            normalize=False,
            dtype=dtype
        )
