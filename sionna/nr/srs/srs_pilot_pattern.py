"""
srs_pilot_pattern.py

Definition of the SRS pilot pattern class that generates uplink SRS pilot signals.
The SRSPilotPattern inherits from sionna.ofdm.PilotPattern and creates both a pilot
mask and corresponding pilot symbols based on one or more SRSConfig objects.
"""

import warnings
import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern

from srs_config import SRSConfig


def zadoff_chu_seq(u, Nzc):
    """Generate a Zadoff–Chu sequence of length Nzc with root u.

    Parameters
    ----------
    u : int
        Root index.
    Nzc : int
        Sequence length.

    Returns
    -------
    seq : np.ndarray
        Zadoff–Chu sequence (complex64).
    """
    n = np.arange(Nzc)
    seq = np.exp(-1j * np.pi * u * n * (n + 1) / float(Nzc))
    return seq.astype(np.complex64)


def generate_srs_sequence(srs_cfg, Msc, num_symbols, port_index):
    r"""Generate an SRS sequence for one port over a number of OFDM symbols.

    Parameters
    ----------
    srs_cfg : SRSConfig
        SRS configuration object.
    Msc : int
        Number of subcarriers allocated to SRS per OFDM symbol.
    num_symbols : int
        Number of OFDM symbols carrying SRS.
    port_index : int
        Index of the antenna port (0-indexed).

    Returns
    -------
    seq_total : np.ndarray
        Concatenated SRS sequence of length num_symbols * Msc.
    """
    # Use NSRSID (offset by port index) as the ZC root (ensure nonzero)
    u = (srs_cfg.NSRSID + port_index) % Msc
    if u == 0:
        u = 1
    seq_list = []
    for _ in range(num_symbols):
        seq = zadoff_chu_seq(u, Msc)
        # Apply cyclic shift: a simple phase rotation across subcarriers.
        n = np.arange(Msc)
        cs_factor = np.exp(-1j * 2 * np.pi * srs_cfg.CyclicShift * n / float(Msc))
        seq_shifted = seq * cs_factor
        seq_list.append(seq_shifted)
    seq_total = np.concatenate(seq_list, axis=0)
    return seq_total


class SRSPilotPattern(PilotPattern):
    r"""SRSPilotPattern(srs_configs, resource_grid, dtype=tf.complex64)

    Creates an SRS pilot pattern that designates the positions and values of SRS signals.
    One or more SRSConfig objects are used (one per user/transmitter) together with a
    ResourceGrid to compute a boolean mask and corresponding pilot symbols.

    Parameters
    ----------
    srs_configs : SRSConfig or list of SRSConfig
        A single SRS configuration object or a list (for multiuser scenarios).
    resource_grid : ResourceGrid
        An instance of sionna.ofdm.ResourceGrid.
    dtype : tf.DType, optional
        Data type for internal calculations (default: tf.complex64).
    """
    def __init__(self,
                 srs_configs,
                 resource_grid,
                 dtype=tf.complex64):
        # Wrap a single configuration into a list.
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        self._srs_configs = srs_configs

        # Get resource grid parameters.
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_eff_sc = resource_grid.num_effective_subcarriers

        # Number of transmitters equals the number of SRS configurations.
        num_tx = len(srs_configs)
        # Assume each SRSConfig defines the number of ports for that user.
        num_streams_per_tx = srs_configs[0].NumSRSPorts

        # Create a boolean mask for the full grid with shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols, num_eff_sc]
        mask = np.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols, num_eff_sc], dtype=bool)

        # List to hold pilot symbols for each transmitter.
        pilots_list = []

        for i, srs_cfg in enumerate(srs_configs):
            # Use the provided Msc.
            Msc = srs_cfg.Msc
            # Determine frequency indices for SRS.
            f_start = srs_cfg.FrequencyStart
            f_end = f_start + Msc
            if f_end > num_eff_sc:
                warnings.warn("SRS frequency block exceeds effective subcarriers; clipping.")
                f_end = num_eff_sc
                Msc = f_end - f_start

            # Determine time indices for SRS.
            t_start = srs_cfg.SymbolStart
            t_end = t_start + srs_cfg.NumSRSSymbols
            if t_end > num_ofdm_symbols:
                raise ValueError("SRS time allocation exceeds available OFDM symbols.")

            # Mark the SRS positions in the mask for every port.
            mask[i, :, t_start:t_end, f_start:f_end] = True

            # Total number of pilot REs per port.
            num_pilots = srs_cfg.NumSRSSymbols * Msc
            pilots_per_user = np.zeros([num_streams_per_tx, num_pilots], dtype=np.complex64)
            for j in range(num_streams_per_tx):
                pilots_per_user[j, :] = generate_srs_sequence(srs_cfg, Msc, srs_cfg.NumSRSSymbols, j)
            pilots_list.append(pilots_per_user)

        # Stack pilots for all transmitters: shape [num_tx, num_streams_per_tx, num_pilots]
        pilots = np.stack(pilots_list, axis=0)

        super().__init__(mask, pilots, trainable=False, normalize=False, dtype=dtype)

    @property
    def srs_configs(self):
        """List of SRS configuration objects."""
        return self._srs_configs
