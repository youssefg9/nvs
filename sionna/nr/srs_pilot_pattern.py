# srs_pilot_pattern.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS pilot pattern for the nr (5G) sub-package of the Sionna library.

This class builds a PilotPattern from one or more SRSConfig objects, with
detailed debug prints to identify shape mismatches and out-of-bounds errors.
"""

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig

__all__ = ["SRSPilotPattern"]

class SRSPilotPattern(PilotPattern):
    r"""
    Builds a pilot pattern for SRS signals from one or more transmitters,
    each defined by an ``SRSConfig``.

    The code includes debug statements that print detailed shape information
    for each transmitter's grid, mask, and pilot extraction step. If an
    IndexError occurs, the code prints the shapes and re-raises the exception,
    helping to locate the problem.

    Parameters
    ----------
    srs_configs : list of SRSConfig
        A list of SRS configuration objects, one per transmitter.

    dtype : tf.DType
        Datatype for internal calculations and the final pattern. Defaults to
        ``tf.complex64``.
    """
    def __init__(self, srs_configs, dtype=tf.complex64):

        # --- Convert single config to a list, check types ---
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        for i, cfg in enumerate(srs_configs):
            if not isinstance(cfg, SRSConfig):
                raise TypeError(f"Element {i} of srs_configs is not an SRSConfig.")

        num_tx = len(srs_configs)
        carrier = srs_configs[0].carrier
        num_sc = carrier.n_size_grid * 12
        num_sym = carrier.num_symbols_per_slot

        # --- Validate each config's carrier dimension ---
        for i, cfg in enumerate(srs_configs):
            if cfg.carrier.num_symbols_per_slot != num_sym:
                raise ValueError(
                    f"[DEBUG] TX={i}: srs_config has num_symbols_per_slot="
                    f"{cfg.carrier.num_symbols_per_slot}, but reference is "
                    f"{num_sym}. Shapes won't align."
                )
            if cfg.carrier.n_size_grid * 12 != num_sc:
                raise ValueError(
                    f"[DEBUG] TX={i}: srs_config has subcarrier dimension="
                    f"{cfg.carrier.n_size_grid*12}, but reference is {num_sc}."
                )

        # --- Determine max number of ports ---
        max_ports = max(cfg.num_srs_ports for cfg in srs_configs)
        print(f"[DEBUG] Found num_tx={num_tx}, max_ports={max_ports}, "
              f"num_sc={num_sc}, num_sym={num_sym}.")

        # --- Allocate mask & pilot arrays in nested structure ---
        mask_all = np.zeros((num_tx, max_ports, num_sym, num_sc), dtype=bool)
        pilot_values = [[None]*max_ports for _ in range(num_tx)]

        # =============== MAIN LOOP ===============
        for tx in range(num_tx):
            cfg = srs_configs[tx]
            print(f"\n[DEBUG] Processing TX={tx} with num_srs_ports={cfg.num_srs_ports}...")

            # -- 1) Get SRS mask and transpose --
            srs_mask_np = np.asarray(cfg.srs_mask())
            print(f"[DEBUG] TX={tx}: srs_mask initial shape = {srs_mask_np.shape} "
                  f"(should be (num_sc, num_sym))")
            if srs_mask_np.shape[0] != num_sc or srs_mask_np.shape[1] != num_sym:
                print(f"[DEBUG] TX={tx}: srs_mask shape mismatch: {srs_mask_np.shape} vs. "
                      f"({num_sc}, {num_sym})")
                raise ValueError(f"SRS mask shape mismatch for TX={tx}.")
            srs_mask_np = srs_mask_np.T  # shape => (num_sym, num_sc)
            print(f"[DEBUG] TX={tx}: srs_mask after transpose => {srs_mask_np.shape}.")

            # -- 2) Get SRS grid and ensure it's a 3D array of shape (ports, sc, sym) --
            grid = np.asarray(cfg.srs_grid)
            print(f"[DEBUG] TX={tx}: srs_grid original shape = {grid.shape}.")

            # Force 3D
            if grid.ndim == 2:
                # Assume shape => (num_sc, num_sym)
                print(f"[DEBUG] TX={tx}: Found 2D grid => adding new axis for ports.")
                grid = grid.reshape((1, grid.shape[0], grid.shape[1]))
            elif grid.ndim > 3:
                raise ValueError(f"[DEBUG] TX={tx}: srs_grid has >3 dims => shape={grid.shape}.")

            # If the first axis isn't equal to cfg.num_srs_ports, try resizing
            if grid.shape[0] != cfg.num_srs_ports:
                print(f"[DEBUG] TX={tx}: grid ports dimension = {grid.shape[0]}, "
                      f"expected {cfg.num_srs_ports} => resizing/replicating.")
                # We'll replicate data using np.resize (which repeats memory).
                # shape => (cfg.num_srs_ports, num_sc, num_sym)
                grid = np.resize(grid, (cfg.num_srs_ports, num_sc, num_sym))
                print(f"[DEBUG] TX={tx}: grid after resize => shape {grid.shape}.")

            # Final shape check
            if grid.shape != (cfg.num_srs_ports, num_sc, num_sym):
                raise ValueError(f"[DEBUG] TX={tx}: srs_grid final shape mismatch. "
                                 f"Got {grid.shape}, expected "
                                 f"({cfg.num_srs_ports}, {num_sc}, {num_sym}).")

            # -- 3) Transpose to (num_sym, num_sc, ports) --
            grid_t = np.transpose(grid, [2, 1, 0])
            print(f"[DEBUG] TX={tx}: grid_t shape = {grid_t.shape} => "
                  f"(num_sym={num_sym}, num_sc={num_sc}, ports={cfg.num_srs_ports})")

            # -- 4) Fill mask & extract pilot values for each port in this config --
            try:
                for p in range(cfg.num_srs_ports):
                    mask_all[tx, p, :, :] = srs_mask_np
                    # Flatten the relevant slices
                    grid_port_flat = grid_t[:, :, p].flatten(order='C')
                    mask_flat = srs_mask_np.flatten(order='C')
                    # pilot vals are the subset
                    pilot_seq = grid_port_flat[mask_flat]
                    pilot_values[tx][p] = pilot_seq
                    print(f"[DEBUG] TX={tx}, port={p}: pilot_seq length={len(pilot_seq)}.")
            except IndexError as e:
                # Print debug shapes if an IndexError occurs
                print(f"[DEBUG] TX={tx}, in port loop => IndexError!")
                print(f"[DEBUG] grid_t.shape={grid_t.shape}, srs_mask_np.shape={srs_mask_np.shape},"
                      f" p={p}, cfg.num_srs_ports={cfg.num_srs_ports}")
                raise e  # re-raise the error

            # For any leftover ports up to max_ports, fill with None for now
            for p in range(cfg.num_srs_ports, max_ports):
                pilot_values[tx][p] = None
                print(f"[DEBUG] TX={tx}, missing port {p} => assigned None pilot.")

        # =============== Build the uniform pilot array ===============
        # Find global max length
        global_max_len = 0
        for tx in range(num_tx):
            for p in range(max_ports):
                vals = pilot_values[tx][p]
                if vals is not None:
                    global_max_len = max(global_max_len, len(vals))
        print(f"\n[DEBUG] global_max_len of all pilot sequences = {global_max_len}.")

        # For missing streams, fill with zeros
        for tx in range(num_tx):
            for p in range(max_ports):
                if pilot_values[tx][p] is None:
                    pilot_values[tx][p] = np.zeros(global_max_len, dtype=complex)
                else:
                    # If sequence is shorter, we won't forcibly expand it,
                    # but it should be fine as is. We'll store in the final array below.
                    pass

        # Create final pilots array
        pilots_all = np.zeros((num_tx, max_ports, global_max_len), dtype=complex)
        for tx in range(num_tx):
            for p in range(max_ports):
                seq = pilot_values[tx][p]
                pilots_all[tx, p, :len(seq)] = seq

        # Debug prints
        print(f"[DEBUG] final mask_all shape => {mask_all.shape}")
        print(f"[DEBUG] final pilots_all shape => {pilots_all.shape}")

        # -- finalize by calling the base PilotPattern constructor
        super().__init__(
            mask_all,
            pilots_all,
            trainable=False,
            normalize=False,
            dtype=dtype
        )
        print("[DEBUG] SRSPilotPattern successfully created.\n")
