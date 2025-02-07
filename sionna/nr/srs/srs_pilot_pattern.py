# srs_pilot_pattern.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS pilot pattern for the nr (5G) sub-package of the Sionna library.

This class builds a PilotPattern from one or more SRSConfig objects. It assumes
that each SRSConfig produces the same number of pilot REs. Optionally, a base
pilot pattern (e.g. for PUSCH-DMRS) can be provided, in which case the SRS pilots
are merged with the base pattern (provided the two masks are disjoint).

Example:
    carrier = CarrierConfig(...)
    srs1 = SRSConfig(carrier)
    srs2 = SRSConfig(carrier)
    # To use SRS only:
    pattern = SRSPilotPattern([srs1, srs2])
    # To merge with an existing DMRS pilot pattern:
    pattern = SRSPilotPattern([srs1, srs2], base_pattern=dmrs_pattern)
"""

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig

__all__ = ["SRSPilotPattern"]

class SRSPilotPattern(PilotPattern):
    def __init__(self, srs_configs, dtype=tf.complex64, base_pattern=None):
        # Convert a single config to a list.
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]

        # Basic type checks.
        for i, cfg in enumerate(srs_configs):
            if not isinstance(cfg, SRSConfig):
                raise TypeError(f"srs_configs[{i}] is not an SRSConfig.")

        num_tx = len(srs_configs)
        carrier = srs_configs[0].carrier
        num_sc = carrier.n_size_grid * 12
        num_sym = carrier.num_symbols_per_slot

        # Determine maximum number of ports among SRS configs.
        max_ports = max(cfg.num_srs_ports for cfg in srs_configs)

        # Prepare arrays for the SRS-only pattern.
        # mask_all: [num_tx, max_ports, num_sym, num_sc]
        mask_all = np.zeros((num_tx, max_ports, num_sym, num_sc), dtype=bool)
        pilot_list = [[None] * max_ports for _ in range(num_tx)]
        pilot_counts = []
        for tx in range(num_tx):
            cfg = srs_configs[tx]
            # srs_mask: shape (num_sc, num_sym), then transpose to (num_sym, num_sc)
            mask_sc_sym = cfg.srs_mask()
            if mask_sc_sym.shape != (num_sc, num_sym):
                raise ValueError(f"TX={tx}: srs_mask shape mismatch. Expected {(num_sc, num_sym)}, got {mask_sc_sym.shape}")
            mask_sym_sc = mask_sc_sym.T
            # srs_grid: shape (num_srs_ports, num_sc, num_sym); if 2D, reshape to (1, ...)
            grid = cfg.srs_grid
            if grid.ndim == 2:
                grid = np.reshape(grid, (1, grid.shape[0], grid.shape[1]))
            if grid.shape[0] != cfg.num_srs_ports or grid.shape[1] != num_sc or grid.shape[2] != num_sym:
                raise ValueError(f"TX={tx}: srs_grid shape mismatch. Got {grid.shape}, expected {(cfg.num_srs_ports, num_sc, num_sym)}")
            grid_t = np.transpose(grid, [2, 1, 0])  # => (num_sym, num_sc, num_ports)
            for p in range(cfg.num_srs_ports):
                mask_all[tx, p] = mask_sym_sc
                grid_flat = grid_t[:, :, p].ravel(order='C')
                mask_flat = mask_sym_sc.ravel(order='C')
                pilot_seq = grid_flat[mask_flat]
                pilot_list[tx][p] = pilot_seq
                pilot_counts.append(len(pilot_seq))
            for p in range(cfg.num_srs_ports, max_ports):
                mask_all[tx, p] = False
                pilot_list[tx][p] = np.zeros(0, dtype=complex)
                pilot_counts.append(0)

        if len(set(pilot_counts)) != 1:
            raise ValueError(
                "Different SRS configurations produce different #pilot REs. "
                f"Found pilot counts: {sorted(set(pilot_counts))}."
            )
        n_pilot_srs = pilot_counts[0]

        # If a base pilot pattern is provided, merge it with the SRS pattern.
        if base_pattern is not None:
            # Retrieve base mask and pilots.
            base_mask = base_pattern.mask.numpy() if isinstance(base_pattern.mask, tf.Tensor) else base_pattern.mask
            base_pilots = base_pattern.pilots.numpy() if isinstance(base_pattern.pilots, tf.Tensor) else base_pattern.pilots
            if base_mask.shape != (num_tx, max_ports, num_sym, num_sc):
                raise ValueError("Base pattern mask shape does not match SRS grid dimensions.")
            # Check that base pilot counts are uniform.
            base_pilot_counts = []
            for tx in range(num_tx):
                for p in range(max_ports):
                    base_flat = base_mask[tx, p].ravel(order='C')
                    count = np.sum(base_mask[tx, p])
                    base_pilot_counts.append(count)
            if len(set(base_pilot_counts)) != 1:
                raise ValueError("Base pattern pilot counts are not uniform.")
            n_pilot_base = base_pilot_counts[0]
            # Ensure the two masks are disjoint.
            if np.any(np.logical_and(base_mask, mask_all)):
                raise ValueError("Overlap detected between base pilot pattern and SRS pilot pattern.")
            # Combined mask is the logical OR.
            combined_mask = np.logical_or(base_mask, mask_all)
            combined_n_pilot = n_pilot_base + n_pilot_srs
            combined_pilots = np.zeros((num_tx, max_ports, combined_n_pilot), dtype=complex)
            # For each tx and port, combine the pilot sequences in the order defined by the combined mask.
            for tx in range(num_tx):
                for p in range(max_ports):
                    base_mask_flat = base_mask[tx, p].ravel(order='C')
                    srs_mask_flat = mask_all[tx, p].ravel(order='C')
                    combined_mask_flat = combined_mask[tx, p].ravel(order='C')
                    base_seq = base_pilots[tx, p]
                    srs_seq = pilot_list[tx][p]
                    combined_seq = []
                    base_idx = 0
                    srs_idx = 0
                    for flag, b_flag, s_flag in zip(combined_mask_flat, base_mask_flat, srs_mask_flat):
                        if flag:
                            if b_flag:
                                combined_seq.append(base_seq[base_idx])
                                base_idx += 1
                            elif s_flag:
                                combined_seq.append(srs_seq[srs_idx])
                                srs_idx += 1
                    combined_seq = np.array(combined_seq, dtype=complex)
                    if len(combined_seq) != combined_n_pilot:
                        raise ValueError("Combined pilot sequence length mismatch.")
                    combined_pilots[tx, p] = combined_seq
            final_mask = combined_mask
            final_pilots = combined_pilots
        else:
            final_mask = mask_all
            final_pilots = np.zeros((num_tx, max_ports, n_pilot_srs), dtype=complex)
            for tx in range(num_tx):
                for p in range(max_ports):
                    final_pilots[tx, p, :len(pilot_list[tx][p])] = pilot_list[tx][p]

        # Call the base constructor.
        super().__init__(
            final_mask, final_pilots,
            trainable=False,
            normalize=False,
            dtype=dtype
        )
