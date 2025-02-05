# srs_pilot_pattern.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS pilot pattern for the nr (5G) sub-package of the Sionna library.

This class builds a PilotPattern from one or more SRSConfig objects, assuming
they *already* have the same number of pilot REs. No dummy data is added.
If a mismatch occurs, an error is raised.
"""

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig

__all__ = ["SRSPilotPattern"]

class SRSPilotPattern(PilotPattern):
    r"""
    Builds a pilot pattern for multiple SRS configurations, each representing
    one transmitter. **No dummy/padding** is added. If the number of pilot REs
    differs among the transmitters/ports, an error is raised.

    Parameters
    ----------
    srs_configs : list[SRSConfig] or SRSConfig
        A single or list of SRS configurations, one per transmitter.

    dtype : tf.DType, optional
        Datatype for internal calculations and the final pilot pattern
        (defaults to `tf.complex64`).


    Notes
    -----
    - This code checks that *all* `(tx, port)` pairs produce the **same**
      number of pilot REs in the mask. If not, an exception is raised.  
    - No debug prints or dummy overhead is introduced.
    - This satisfies Sionna’s requirement that
      “The number of nonzero elements in the masks for all transmitters
      and streams must be identical.”

    Example
    -------
    .. code-block:: python

       carrier = CarrierConfig(...)
       srs1 = SRSConfig(carrier)
       srs2 = SRSConfig(carrier)
       # Must ensure srs1, srs2 produce the same # pilot REs
       pattern = SRSPilotPattern([srs1, srs2])
    """
    def __init__(self, srs_configs, dtype=tf.complex64):

        # Convert single config to list
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]

        # Basic checks
        for i, cfg in enumerate(srs_configs):
            if not isinstance(cfg, SRSConfig):
                raise TypeError(f"srs_configs[{i}] is not an SRSConfig.")

        num_tx = len(srs_configs)
        carrier = srs_configs[0].carrier
        num_sc = carrier.n_size_grid * 12
        num_sym = carrier.num_symbols_per_slot

        # Validate each config's dimension
        for i, cfg in enumerate(srs_configs):
            if cfg.carrier.n_size_grid * 12 != num_sc:
                raise ValueError(f"SRS config {i} mismatch in subcarrier dimension.")
            if cfg.carrier.num_symbols_per_slot != num_sym:
                raise ValueError(f"SRS config {i} mismatch in number of OFDM symbols.")

        # Determine maximum number of ports among configs
        max_ports = max(cfg.num_srs_ports for cfg in srs_configs)

        # Prepare final arrays
        # shape => mask: [num_tx, max_ports, num_sym, num_sc]
        # shape => pilots: [num_tx, max_ports, ???] => we will unify the pilot count below
        mask_all = np.zeros((num_tx, max_ports, num_sym, num_sc), dtype=bool)
        pilot_list = [[None]*max_ports for _ in range(num_tx)]  # store pilot sequences

        # Gather #pilot RE for each (tx, port)
        pilot_counts = []  # list of ints
        for tx in range(num_tx):
            cfg = srs_configs[tx]
            # srs_mask => shape => (num_subcarriers, num_symbols), then transpose
            mask_sc_sym = cfg.srs_mask()
            if mask_sc_sym.shape != (num_sc, num_sym):
                raise ValueError(f"TX={tx}: srs_mask shape mismatch. "
                                 f"Expected {(num_sc, num_sym)}, got {mask_sc_sym.shape}")
            mask_sym_sc = mask_sc_sym.T  # shape => (num_sym, num_sc)

            # srs_grid => shape => (num_srs_ports, num_sc, num_sym)
            grid = cfg.srs_grid
            if grid.ndim == 2:
                # If it's 2D, we assume 1 port
                grid = np.reshape(grid, (1, grid.shape[0], grid.shape[1]))
            if grid.shape[0] != cfg.num_srs_ports or grid.shape[1] != num_sc or grid.shape[2] != num_sym:
                raise ValueError(f"TX={tx}: srs_grid shape mismatch. "
                                 f"Got {grid.shape}, expected {(cfg.num_srs_ports, num_sc, num_sym)}")

            grid_t = np.transpose(grid, [2,1,0])  # => (num_sym, num_sc, num_ports)

            for p in range(cfg.num_srs_ports):
                # fill mask
                mask_all[tx, p] = mask_sym_sc
                # flatten pilot
                grid_flat = grid_t[:,:,p].ravel(order='C')
                mask_flat = mask_sym_sc.ravel(order='C')
                pilot_seq = grid_flat[mask_flat]
                pilot_list[tx][p] = pilot_seq
                pilot_counts.append(len(pilot_seq))

            # Ports > cfg.num_srs_ports => empty
            for p in range(cfg.num_srs_ports, max_ports):
                # No usage
                mask_all[tx, p] = False
                pilot_list[tx][p] = np.zeros(0, dtype=complex)
                pilot_counts.append(0)

        # Now check if pilot_counts are identical
        if len(set(pilot_counts)) != 1:
            # They differ => we raise an error
            raise ValueError(
                "Different SRS configurations produce different #pilot REs. "
                "Sionna requires the same number of pilot REs for all transmitters/ports. "
                f"Found pilot counts: {sorted(set(pilot_counts))}. "
                "Cannot unify them in one PilotPattern without dummy data."
            )

        # pilot_counts is uniform => let's define n_pilot
        n_pilot = pilot_counts[0]

        # Build final pilot array => shape => [num_tx, max_ports, n_pilot]
        pilots_all = np.zeros((num_tx, max_ports, n_pilot), dtype=complex)
        for tx in range(num_tx):
            for p in range(max_ports):
                seq = pilot_list[tx][p]
                pilots_all[tx, p, :len(seq)] = seq

        # Call base constructor
        super().__init__(
            mask_all, pilots_all,
            trainable=False,
            normalize=False,
            dtype=dtype
        )
        # Done
