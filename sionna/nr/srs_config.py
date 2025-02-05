# srs_config.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS configuration for the nr (5G) sub-package of the Sionna library.
"""

import numpy as np
from sionna import nr
from .config import Config
from .utils import generate_prng_seq  # This function should generate a pseudo-random bit sequence

__all__ = ["SRSConfig"]

class SRSConfig(Config):
    r"""
    The SRSConfig object holds parameters for an uplink sounding reference
    signal (SRS), as defined in 3GPP 38.211 §6.4.1.4.

    It provides methods to compute:

      - Which slots are candidates for SRS transmission
      - The actual SRS symbols (here, a simple QPSK mapping is used for demonstration)
      - An SRS mask for the subcarriers/symbols in the slot

    Parameters
    ----------
    carrier_config : sionna.nr.CarrierConfig or None
        Defines the carrier settings (subcarrier spacing, grid size, etc.).

    Example
    -------
    >>> srs_config = SRSConfig()
    >>> srs_config.num_srs_ports = 2
    >>> srs_config.num_srs_symbols = 4
    >>> srs_grid = srs_config.srs_grid
    """

    def __init__(self, carrier_config=None, **kwargs):
        super().__init__(**kwargs)

        self._name = "SRS Configuration"
        self.carrier = carrier_config

        # Default or user-supplied values
        self._ifndef("num_srs_ports", 1)
        self._ifndef("symbol_start", 0)
        self._ifndef("num_srs_symbols", 1)

        # SRS Period: 'on', 'off', or [T_srs, offset]
        self._ifndef("srs_period", "on")

        # SRS BW configuration indices (Table 6.4.1.4.3-1)
        self._ifndef("c_srs", 0)
        self._ifndef("b_srs", 0)

        # Comb factor K_TC in {2,4,8}
        self._ifndef("k_tc", 2)

        # Group/sequence hopping config: 'neither','groupHopping','sequenceHopping'
        self._ifndef("group_seq_hopping", "neither")

        # SRS Scrambling ID
        self._ifndef("n_srs_id", 0)

        # Partial frequency sounding factor
        self._ifndef("frequency_scaling_factor", 1)

        # Cyclic shift and hopping parameters
        self._ifndef("cyclic_shift", 0)
        self._ifndef("cyclic_shift_hopping", False)
        self._ifndef("cyclic_shift_hopping_id", 0)
        self._ifndef("cyclic_shift_hopping_subset", None)
        self._ifndef("hopping_finer_granularity", False)

        # 8-port TDM flag
        self._ifndef("enable_eight_port_tdm", False)

        # Validate configuration
        self.check_config()

    # === Configurable Properties ===

    @property
    def carrier(self):
        return self._carrier

    @carrier.setter
    def carrier(self, value):
        if value is None:
            value = nr.CarrierConfig()
        else:
            assert isinstance(value, nr.CarrierConfig), \
                "carrier must be an instance of sionna.nr.CarrierConfig"
        self._carrier = value

    @property
    def num_srs_ports(self):
        return self._num_srs_ports

    @num_srs_ports.setter
    def num_srs_ports(self, value):
        # Typically 1, 2, 4, or 8 for SRS
        assert value in [1, 2, 4, 8], "num_srs_ports must be in [1,2,4,8]"
        self._num_srs_ports = value

    @property
    def symbol_start(self):
        return self._symbol_start

    @symbol_start.setter
    def symbol_start(self, value):
        self._symbol_start = value

    @property
    def num_srs_symbols(self):
        return self._num_srs_symbols

    @num_srs_symbols.setter
    def num_srs_symbols(self, value):
        self._num_srs_symbols = value

    @property
    def srs_period(self):
        r"""
        'on', 'off', or a tuple/list [T_srs, offset].
        """
        return self._srs_period

    @srs_period.setter
    def srs_period(self, value):
        if isinstance(value, str):
            assert value in ["on", "off"], "srs_period must be 'on', 'off', or [T_srs, offset]"
        else:
            assert len(value) == 2, "srs_period must be 'on','off' or [T_srs, offset]"
        self._srs_period = value

    @property
    def c_srs(self):
        return self._c_srs

    @c_srs.setter
    def c_srs(self, value):
        assert 0 <= value <= 63, "c_srs must be in [0..63]"
        self._c_srs = value

    @property
    def b_srs(self):
        return self._b_srs

    @b_srs.setter
    def b_srs(self, value):
        assert 0 <= value <= 3, "b_srs must be in [0..3]"
        self._b_srs = value

    @property
    def k_tc(self):
        return self._k_tc

    @k_tc.setter
    def k_tc(self, value):
        assert value in [2, 4, 8], "k_tc must be in [2,4,8]"
        self._k_tc = value

    @property
    def frequency_scaling_factor(self):
        return self._frequency_scaling_factor

    @frequency_scaling_factor.setter
    def frequency_scaling_factor(self, value):
        assert value in [1, 2, 4], "frequency_scaling_factor must be in [1,2,4]"
        self._frequency_scaling_factor = value

    @property
    def cyclic_shift(self):
        return self._cyclic_shift

    @cyclic_shift.setter
    def cyclic_shift(self, value):
        self._cyclic_shift = value

    @property
    def group_seq_hopping(self):
        return self._group_seq_hopping

    @group_seq_hopping.setter
    def group_seq_hopping(self, value):
        assert value in ["neither", "groupHopping", "sequenceHopping"], \
            "group_seq_hopping must be 'neither','groupHopping','sequenceHopping'"
        self._group_seq_hopping = value

    @property
    def n_srs_id(self):
        return self._n_srs_id

    @n_srs_id.setter
    def n_srs_id(self, value):
        assert 0 <= value <= 65535, "n_srs_id must be in [0..65535]"
        self._n_srs_id = value

    @property
    def enable_eight_port_tdm(self):
        return self._enable_eight_port_tdm

    @enable_eight_port_tdm.setter
    def enable_eight_port_tdm(self, value):
        assert isinstance(value, bool), "enable_eight_port_tdm must be bool"
        self._enable_eight_port_tdm = value

    @property
    def cyclic_shift_hopping(self):
        return self._cyclic_shift_hopping

    @cyclic_shift_hopping.setter
    def cyclic_shift_hopping(self, value):
        assert isinstance(value, bool), "cyclic_shift_hopping must be bool"
        self._cyclic_shift_hopping = value

    @property
    def cyclic_shift_hopping_id(self):
        return self._cyclic_shift_hopping_id

    @cyclic_shift_hopping_id.setter
    def cyclic_shift_hopping_id(self, value):
        assert 0 <= value <= 65535, "cyclic_shift_hopping_id must be in [0..65535]"
        self._cyclic_shift_hopping_id = value

    @property
    def cyclic_shift_hopping_subset(self):
        return self._cyclic_shift_hopping_subset

    @cyclic_shift_hopping_subset.setter
    def cyclic_shift_hopping_subset(self, value):
        self._cyclic_shift_hopping_subset = value

    @property
    def hopping_finer_granularity(self):
        return self._hopping_finer_granularity

    @hopping_finer_granularity.setter
    def hopping_finer_granularity(self, value):
        assert isinstance(value, bool), "hopping_finer_granularity must be bool"
        self._hopping_finer_granularity = value

    # === Derived / Read-Only Properties ===

    @property
    def is_candidate_slot(self):
        r"""
        Determines whether the current slot is a candidate for SRS transmission
        based on the srs_period and the current slot/frame in the carrier.
        """
        slot = self.carrier.slot_number
        nframe = self.carrier.frame_number
        if isinstance(self.srs_period, str):
            return self.srs_period == 'on'
        else:
            T_srs, offset = self.srs_period
            total_slot = nframe * self.carrier.num_slots_per_frame + slot
            return ((total_slot - offset) % T_srs == 0)

    @property
    def num_subcarriers(self):
        """
        Number of subcarriers for the entire carrier grid.
        """
        return 12 * self.carrier.n_size_grid

    @property
    def num_symbols_per_slot(self):
        """
        Number of OFDM symbols per slot (e.g., 14 for normal CP).
        """
        return self.carrier.num_symbols_per_slot

    @property
    def m_srs_bsrs(self):
        """
        Returns the number of resource blocks for the [c_srs, b_srs] combination.
        For demonstration, a simple placeholder mapping is used.
        """
        return self._lookup_srs_bandwidth_config(self.c_srs, self.b_srs)

    def _lookup_srs_bandwidth_config(self, c_srs, b_srs):
        """
        Minimal placeholder for the 6.4.1.4.3-1 table.
        Returns a simplified M_srs value.
        """
        M_srs = (c_srs + 1) * 4
        return M_srs

    # === SRS Generation ===

    @property
    def srs_grid(self):
        r"""
        Returns a resource grid of shape:
          [num_srs_ports, num_subcarriers, num_symbols_per_slot]
        filled with SRS signals if the slot is a candidate; otherwise, an all-zero array.
        """
        if not self.is_candidate_slot:
            return np.zeros([self.num_srs_ports,
                             self.num_subcarriers,
                             self.num_symbols_per_slot], dtype=complex)

        A = np.zeros([self.num_srs_ports,
                      self.num_subcarriers,
                      self.num_symbols_per_slot], dtype=complex)
        M_srs_all = self.m_srs_bsrs
        n_sub_srs = (M_srs_all * 12) // (self.k_tc * self.frequency_scaling_factor)
        start_sym = self.symbol_start
        # For each SRS symbol (within the allocated region)
        for s in range(self.num_srs_symbols):
            l = start_sym + s
            c_init = self._c_init_for_srs(l)
            num_bits = 2 * n_sub_srs
            c_seq = generate_prng_seq(num_bits, c_init=c_init)
            r_seq = (1 / np.sqrt(2)) * ((1 - 2 * c_seq[::2]) + 1j * (1 - 2 * c_seq[1::2]))
            r_port = np.zeros([self.num_srs_ports, n_sub_srs], dtype=complex)
            for p in range(self.num_srs_ports):
                r_port[p] = r_seq  # Here one could add port-dependent phase shifts.
            for p in range(self.num_srs_ports):
                A[p, 0:n_sub_srs, l] = r_port[p]
        return A

    def _c_init_for_srs(self, sym_idx):
        """
        Computes a simple PRBS initialization value for SRS generation.
        """
        slot_number = self.carrier.slot_number
        n_id = self.n_srs_id
        c_init = (2**17 * (slot_number + sym_idx + 1) * (2 * n_id + 1)) % (2**31)
        return int(c_init)

    def srs_mask(self):
        r"""
        Returns a boolean mask of shape [num_subcarriers, num_symbols_per_slot]
        that marks the resource elements (REs) used for SRS.
        """
        M_srs_all = self.m_srs_bsrs
        n_sub_srs = (M_srs_all * 12) // (self.k_tc * self.frequency_scaling_factor)
        mask = np.zeros([self.num_subcarriers, self.num_symbols_per_slot], dtype=bool)
        if self.is_candidate_slot:
            s0 = self.symbol_start
            s1 = s0 + self.num_srs_symbols
            mask[0:n_sub_srs, s0:s1] = True
        return mask

    def check_config(self):
        """
        Validates the SRS configuration.
        """
        self.carrier.check_config()
        assert (self.symbol_start + self.num_srs_symbols) <= self.carrier.num_symbols_per_slot, \
            "Symbol allocation invalid for the given CP length."
        return True

    def show(self):
        """Prints the SRS configuration."""
        print("=== SRS Configuration ===")
        super().show()
        self.carrier.show()
