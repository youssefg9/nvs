# srs_pilot_pattern.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS pilot pattern for the nr (5G) sub-package of the Sionna library, with
a workaround to unify the number of pilot REs across all transmitters/ports.
"""

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig

__all__ = ["SRSPilotPattern"]

class SRSPilotPattern(PilotPattern):
    r"""
    Builds a pilot pattern for multiple SRS configurations, **forcing** the
    same pilot count for each transmitter and port by adding dummy pilot REs
    to smaller SRS masks.

    Parameters
    ----------
    srs_configs : list of SRSConfig
        One SRS configuration object per transmitter.

    dtype : tf.DType, optional
        Datatype for internal calculations. Defaults to `tf.complex64`.

    Notes
    -----
    - If some SRS have fewer REs than others, we artificially add REs to
      the smaller ones so that *all* `(tx, port)` ends up with the same number
      of nonzero REs. The added REs are “dummy” with zero pilot values.
    - This satisfies Sionna’s requirement that
      “The number of nonzero elements in the masks for all transmitters and
      streams must be identical.” but physically means we are
      “claiming” pilot in extra REs for the smaller SRS.

    Example
    -------
    .. code-block:: python

       carrier = CarrierConfig()
       srs1 = SRSConfig(carrier_config=carrier, ...)
       srs2 = SRSConfig(carrier_config=carrier, ...)
       pilot_pattern = SRSPilotPattern([srs1, srs2])
    """
    def __init__(self, srs_configs, dtype=tf.complex64):

        # Convert single SRSConfig to list
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        else:
            for i, cfg in enumerate(srs_configs):
                if not isinstance(cfg, SRSConfig):
                    raise TypeError(f"Element {i} of srs_configs is not an SRSConfig.")

        num_tx = len(srs_configs)
        carrier = srs_configs[0].carrier
        num_sc = carrier.n_size_grid * 12
        num_sym = carrier.num_symbols_per_slot

        # Validate each config matches the main carrier dimension
        for i, cfg in enumerate(srs_configs):
            if cfg.carrier.num_symbols_per_slot != num_sym:
                raise ValueError(
                    f"SRS config {i} mismatch in num_symbols_per_slot: "
                    f"{cfg.carrier.num_symbols_per_slot} vs. {num_sym}"
                )
            if cfg.carrier.n_size_grid*12 != num_sc:
                raise ValueError(
                    f"SRS config {i} mismatch in subcarrier dimension: "
                    f"{cfg.carrier.n_size_grid*12} vs. {num_sc}"
                )

        # Determine maximum number of ports
        max_ports = max(cfg.num_srs_ports for cfg in srs_configs)

        # Allocate final mask shape => [num_tx, max_ports, num_sym, num_sc]
        # We'll fill these after unifying the per-port masks
        mask_all = np.zeros((num_tx, max_ports, num_sym, num_sc), dtype=bool)
        # We'll store the pilot sequences in pilot_values[tx][port] before building final array
        pilot_values = [[None]*max_ports for _ in range(num_tx)]

        # Step 1) Collect the actual (row,col) = (symbol, subcarrier) coords used by each (tx,port)
        # We also store the pilot values for those coords.
        # Then we unify the counts by adding dummy coords if needed.

        # For gathering info
        all_masks = []  # list of shape => [ (tx,port, srs_mask) ]
        all_pilots = [] # parallel structure => [ (tx,port, pilot_values_for_that_port) ]

        for tx in range(num_tx):
            cfg = srs_configs[tx]
            srs_mask = cfg.srs_mask().T  # shape => (num_sym, num_sc)
            # Flatten the mask to find the True coords
            coords = np.where(srs_mask)  # coords is (2, #True)
            # srs_grid => shape => (cfg.num_srs_ports, num_sc, num_sym)
            srs_grid = np.asarray(cfg.srs_grid)
            if srs_grid.ndim == 2:
                srs_grid = srs_grid.reshape((1, srs_grid.shape[0], srs_grid.shape[1]))
            # If first axis != num_srs_ports, resize
            if srs_grid.shape[0] != cfg.num_srs_ports:
                srs_grid = np.resize(srs_grid, (cfg.num_srs_ports, num_sc, num_sym))

            # Transpose => shape => (num_sym, num_sc, ports)
            grid_t = np.transpose(srs_grid, [2,1,0])

            for p in range(cfg.num_srs_ports):
                # Collect coords for pilot
                pilot_seq = grid_t[coords[0], coords[1], p]  # fancy indexing
                # Store
                pilot_values[tx][p] = pilot_seq
                # We'll store the coords themselves in a separate structure
                all_masks.append((tx, p, coords[0], coords[1]))  # symbol coords in coords[0], subcarrier coords in coords[1]
                all_pilots.append((tx, p, pilot_seq))

            # For ports from cfg.num_srs_ports..max_ports => no usage => fill later with dummy
            for p in range(cfg.num_srs_ports, max_ports):
                pilot_values[tx][p] = None
                # We'll store empty coords => no usage
                all_masks.append((tx, p, np.array([], dtype=int), np.array([], dtype=int)))
                all_pilots.append((tx, p, np.zeros(0, dtype=complex)))

        # Step 2) Find the maximum pilot count across all (tx,port)
        max_npilot = 0
        for tx in range(num_tx):
            for p in range(max_ports):
                seq = pilot_values[tx][p]
                if seq is not None:
                    max_npilot = max(max_npilot, len(seq))
        # E.g. we have max_npilot=288 in your example

        # Step 3) "Pad" the smaller pilot sets so each (tx,port) also has max_npilot REs
        # We'll do so by adding extra RE coords (dummy) anywhere we can. For simplicity,
        # we pick from the subcarrier x symbol region not already used by that (tx,port).
        # We fill pilot values = 0 for those dummy REs.

        # We'll create a final structure that returns the "unified" coords for each (tx,port).
        # Then we can fill a final mask & pilot array of consistent size.

        # Pre-build a list of all possible coords => shape => (num_sym * num_sc, 2)
        # We’ll do (sym, sc) in row-major order
        all_possible_coords = [(s, c) for s in range(num_sym) for c in range(num_sc)]

        # We'll store the new coords
        final_coords = [[None]*max_ports for _ in range(num_tx)]
        final_pilots = [[None]*max_ports for _ in range(num_tx)]

        for (tx, p, sym_coords, sc_coords), (_, _, pilot_seq) in zip(all_masks, all_pilots):
            # We know pilot_seq has length = len(sym_coords)
            # unify => we want length= max_npilot
            needed = max_npilot - len(pilot_seq)
            if needed <= 0:
                # we can just store
                final_coords[tx][p] = (sym_coords, sc_coords)
                final_pilots[tx][p] = pilot_seq
            else:
                # Need dummy. We'll find leftover coords that are not in the current set
                used_set = set(zip(sym_coords, sc_coords))
                leftover = []
                for (symi, sci) in all_possible_coords:
                    if (symi, sci) not in used_set:
                        leftover.append((symi, sci))
                # If leftover is too small => or equal. Usually leftover is large
                # We'll pick the first 'needed' leftover coords
                leftover = leftover[:needed]
                # Build new coords array
                new_sym_coords = np.concatenate([sym_coords, np.array([x[0] for x in leftover], dtype=int)])
                new_sc_coords  = np.concatenate([sc_coords,  np.array([x[1] for x in leftover], dtype=int)])
                # Build new pilot
                new_pilot_seq = np.concatenate([pilot_seq, np.zeros(needed, dtype=complex)])
                # store
                final_coords[tx][p] = (new_sym_coords, new_sc_coords)
                final_pilots[tx][p] = new_pilot_seq

        # Now we have final_coords with the same length (= max_npilot).
        # Step 4) Build the final mask & pilot arrays.

        # We can fill final mask => shape => (num_tx, max_ports, num_sym, num_sc)
        final_mask = np.zeros((num_tx, max_ports, num_sym, num_sc), dtype=bool)
        for tx in range(num_tx):
            for p in range(max_ports):
                (sym_coords, sc_coords) = final_coords[tx][p]
                final_mask[tx, p, sym_coords, sc_coords] = True

        # Build final pilots => shape => (num_tx, max_ports, max_npilot)
        final_pilots_arr = np.zeros((num_tx, max_ports, max_npilot), dtype=complex)
        for tx in range(num_tx):
            for p in range(max_ports):
                final_pilots_arr[tx, p, :] = final_pilots[tx][p]

        # Print debug info
        print(f"[MASK UNIFY] Found max_npilot={max_npilot} across all (tx,port).")
        print(f"[MASK UNIFY] final_mask shape={final_mask.shape}, final_pilots shape={final_pilots_arr.shape}.")
        # Build the PilotPattern
        super().__init__(
            final_mask,
            final_pilots_arr,
            trainable=False,
            normalize=False,
            dtype=dtype
        )
        print("[MASK UNIFY] SRSPilotPattern created successfully. " 
              "All (tx, port) have the same pilot RE count now.\n")
