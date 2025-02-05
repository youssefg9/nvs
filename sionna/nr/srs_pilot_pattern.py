# srs_pilot_pattern.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS pilot pattern for the nr (5G) sub-package of the Sionna library.
"""
import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig

__all__ = ["SRSPilotPattern"]

class SRSPilotPattern(PilotPattern):
    r"""
    Define an OFDM PilotPattern for one or more SRS transmissions.

    Parameters
    ----------
    srs_configs : list of SRSConfig
        A list of SRS configuration objects, one per transmitter.
    
    dtype : tf.DType
        Datatype for internal calculations and final output. Defaults to tf.complex64.

    Notes
    -----
    - The output pilot pattern can be used in Sionna’s OFDM pipeline for channel
      estimation, specifically for the SRS signals in the uplink.
    - If multiple transmitters share overlapping subcarriers and symbols,
      be mindful of collisions or how you want to handle them.
    """
    def __init__(self, srs_configs, dtype=tf.complex64):

        # Handle single or multiple configs
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        else:
            for cfg in srs_configs:
                assert isinstance(cfg, SRSConfig), \
                    "All elements must be SRSConfig instances."

        # Basic checks
        num_tx = len(srs_configs)
        carrier = srs_configs[0].carrier
        num_srs_ports = srs_configs[0].num_srs_ports
        num_sc = carrier.n_size_grid * 12
        num_sym = carrier.num_symbols_per_slot

        # Check all srs configs match the same subcarrier/symbol allocation
        for i, cfg in enumerate(srs_configs):
            assert cfg.carrier.num_symbols_per_slot == num_sym, \
                "All SRS configs must have same num_symbols_per_slot."
            assert (cfg.carrier.n_size_grid * 12) == num_sc, \
                "All SRS configs must have same subcarrier dimension."

        # Create placeholders for the final mask and pilot values:
        #   mask shape = [num_tx, num_srs_ports, num_sym, num_sc]
        #   pilots shape = [num_tx, num_srs_ports, #pilots]
        mask_all = np.zeros([num_tx, num_srs_ports, num_sym, num_sc], dtype=bool)
        # We'll collect all pilot symbols
        # We need to figure out how many total SRS REs are used
        # We’ll do that per config below.
        pilot_list = []

        for tx in range(num_tx):
            cfg = srs_configs[tx]
            # get SRS mask => shape [num_subcarriers, num_symbols]
            srs_mask = cfg.srs_mask().T  # shape => [num_symbols, num_subcarriers]
            # get SRS grid => shape [num_srs_ports, num_subcarriers, num_symbols]
            srs_grid = cfg.srs_grid
            # reorder srs_grid so axis=0 => # of symbols, axis=1 => # subcarriers
            # We had shape (port, subcarriers, symbols); we want (symbols, subcarriers, port)
            # to align with the iteration below
            srs_grid = np.transpose(srs_grid, [2,1,0])  # shape => [symbols, subcarriers, ports]

            for p in range(cfg.num_srs_ports):
                # Mark the mask
                mask_all[tx, p, :, :] = srs_mask
                # extract pilot symbols from srs_grid
                # shape => [symbols, subcarriers]
                srs_vals = srs_grid[:,:,p]
                pilot_vals = srs_vals[srs_mask]
                pilot_list.append((tx,p,pilot_vals))

        # Now we must create the final pilots array in shape [num_tx, num_srs_ports, #pilots]
        # We can do this in a second pass or store them directly.
        # For each (tx, port), we do pilot_all[tx,port,:] = the pilot values
        # We need to ensure consistent ordering. The PilotPattern expects
        # that `mask[i,j] = True` => the pilot in `pilots[i,j,k]`.
        # So we collect them in that order:

        pilots_all = []
        for tx in range(num_tx):
            pilots_ports = []
            for p in range(num_srs_ports):
                # find the matching pilot_list entry
                # or we can do direct indexing
                # but we have to be sure each (tx,p) was appended once
                # Simplify: re-collect from the srs_grid as above
                # The “flatten order” is row-major, so we do the same:
                mp = mask_all[tx,p,:,:].flatten(order='C')
                # srs_grid shape => [sym, sc, ports], but we already have it. We'll re-get:
                srs_grid = srs_configs[tx].srs_grid
                srs_grid = np.transpose(srs_grid, [2,1,0])  # [sym, sc, port]
                srs_vals = srs_grid[:,:,p].flatten(order='C')
                # pick out where mask==True
                pilot_vals = srs_vals[mp]
                pilots_ports.append(pilot_vals)
            pilots_all.append(pilots_ports)

        # Convert to final np array
        # shape => [num_tx, num_srs_ports, #pilots]
        # note that #pilots can vary across tx/ports if they have different SRS usage
        # or the same if consistent. For simplicity, we’ll assume all have same # REs,
        # but if not, you must handle variable length (PilotPattern requires uniform).
        # Let’s assume they do for demonstration:
        max_len = 0
        for tx in range(num_tx):
            for p in range(num_srs_ports):
                max_len = max(max_len, len(pilots_all[tx][p]))

        # We'll store them, zero-padding if needed
        final_pilots = np.zeros([num_tx, num_srs_ports, max_len], dtype=complex)
        for tx in range(num_tx):
            for p in range(num_srs_ports):
                arr = pilots_all[tx][p]
                final_pilots[tx,p,:len(arr)] = arr

        # Finally, init the base PilotPattern
        # mask shape => [num_tx, num_srs_ports, num_sym, num_sc]
        # pilots shape => [num_tx, num_srs_ports, #pilots]
        super().__init__(mask_all,
                         final_pilots,
                         trainable=False,
                         normalize=False,
                         dtype=dtype)
