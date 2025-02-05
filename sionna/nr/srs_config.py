# srs_config.py
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 
# SPDX-License-Identifier: Apache-2.0
#
"""
SRS configuration for the nr (5G) sub-package of the Sionna library.

This module defines the SRSConfig class which holds uplink SRS configuration
parameters (per 3GPP 38.211 §6.4.1.4) and provides methods to generate the
SRS grid and supplementary information.
"""

import numpy as np
from sionna import nr
from .config import Config

__all__ = ["SRSConfig"]

class SRSConfig(Config):
    """
    SRSConfig holds the configuration for an uplink SRS.

    Configurable parameters include:
      - num_srs_ports, symbol_start, num_srs_symbols, srs_period,
      - c_srs, b_srs, k_tc, group_seq_hopping, n_srs_id,
      - frequency_scaling_factor, cyclic_shift, cyclic_shift_hopping,
      - cyclic_shift_hopping_id, cyclic_shift_hopping_subset, hopping_finer_granularity,
      - enable_eight_port_tdm.

    The class computes additional parameters such as the group and sequence
    numbers, cyclic shift (alpha), and returns an SRS grid array of shape
      [num_srs_ports, num_subcarriers, num_symbols_per_slot].
    It also stores an info dictionary (keys: "SeqGroup", "NSeq", "Alpha", "SeqLength").
    """

    def __init__(self, carrier_config=None, **kwargs):
        super().__init__(**kwargs)
        self._name = "SRS Configuration"
        self.carrier = carrier_config

        # Set default values (or use user-supplied ones via kwargs)
        self._ifndef("num_srs_ports", 1)
        self._ifndef("symbol_start", 0)
        self._ifndef("num_srs_symbols", 1)
        self._ifndef("srs_period", "on")  # alternatively, a tuple [T_srs, offset]
        self._ifndef("c_srs", 0)
        self._ifndef("b_srs", 0)
        self._ifndef("k_tc", 2)
        self._ifndef("group_seq_hopping", "neither")  # 'neither', 'groupHopping', 'sequenceHopping'
        self._ifndef("n_srs_id", 0)
        self._ifndef("frequency_scaling_factor", 1)
        self._ifndef("cyclic_shift", 0)
        self._ifndef("cyclic_shift_hopping", False)
        self._ifndef("cyclic_shift_hopping_id", 0)
        self._ifndef("cyclic_shift_hopping_subset", None)
        self._ifndef("hopping_finer_granularity", False)
        self._ifndef("enable_eight_port_tdm", False)

        self.check_config()

    # --- Configurable Properties ---

    @property
    def carrier(self):
        return self._carrier
    @carrier.setter
    def carrier(self, value):
        if value is None:
            value = nr.CarrierConfig()
        else:
            assert isinstance(value, nr.CarrierConfig), "carrier must be an instance of nr.CarrierConfig"
        self._carrier = value

    @property
    def num_srs_ports(self):
        return self._num_srs_ports
    @num_srs_ports.setter
    def num_srs_ports(self, value):
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
    def group_seq_hopping(self):
        return self._group_seq_hopping
    @group_seq_hopping.setter
    def group_seq_hopping(self, value):
        assert value in ["neither", "groupHopping", "sequenceHopping"], "Invalid group_seq_hopping"
        self._group_seq_hopping = value

    @property
    def n_srs_id(self):
        return self._n_srs_id
    @n_srs_id.setter
    def n_srs_id(self, value):
        assert 0 <= value <= 65535, "n_srs_id must be in [0..65535]"
        self._n_srs_id = value

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

    @property
    def enable_eight_port_tdm(self):
        return self._enable_eight_port_tdm
    @enable_eight_port_tdm.setter
    def enable_eight_port_tdm(self, value):
        assert isinstance(value, bool), "enable_eight_port_tdm must be bool"
        self._enable_eight_port_tdm = value

    # --- Derived / Read-only Properties ---

    @property
    def is_candidate_slot(self):
        slot = self.carrier.slot_number
        nframe = self.carrier.frame_number
        if isinstance(self.srs_period, str):
            return self.srs_period == 'on'
        else:
            T_srs, offset = self.srs_period
            total_slot = nframe * self.carrier.num_slots_per_frame + slot
            return ((total_slot - offset) % T_srs) == 0

    @property
    def num_subcarriers(self):
        return 12 * self.carrier.n_size_grid

    @property
    def num_symbols_per_slot(self):
        return self.carrier.num_symbols_per_slot

    @property
    def m_srs_bsrs(self):
        return self._lookup_srs_bandwidth_config(self.c_srs, self.b_srs)

    def _lookup_srs_bandwidth_config(self, c_srs, b_srs):
        # Simplified mapping for demonstration.
        return (c_srs + 1) * 4

    # --- Helper Methods for SRS Generation ---
    @staticmethod
    def _zc_sequence(N, q):
        n = np.arange(N)
        return np.exp(-1j * np.pi * q * n * (n + 1) / N)

    def _group_number_hopping(self):
        return np.mod(self.n_srs_id, 30) * np.ones(self.num_srs_symbols)

    def _sequence_number_hopping(self):
        return np.zeros(self.num_srs_symbols)

    def _get_fcsh(self, nCSmax):
        if self.cyclic_shift_hopping:
            fcsh = np.mod(np.arange(self.num_srs_symbols) + self.cyclic_shift_hopping_id, nCSmax)
            K = 1
        else:
            fcsh = np.zeros(self.num_srs_symbols)
            K = 1
        return fcsh, K

    def _compute_alpha(self, nCSmax):
        NPorts = self.num_srs_ports
        KTC = self.k_tc
        nCS = self.cyclic_shift
        ports8tdm = self.enable_eight_port_tdm

        if KTC == 2:
            nCSmax_local = 8
        elif KTC == 4:
            nCSmax_local = 12
        elif KTC == 8:
            nCSmax_local = 6
        else:
            nCSmax_local = nCSmax

        p = 1000 + np.arange(NPorts)
        if ports8tdm:
            NBarAP = 4
            pBar = 1000 + np.mod(p, 2)
            if NPorts >= 8:
                pBar[4:8] = pBar[4:8] + 2
        else:
            NBarAP = NPorts
            pBar = p

        if NBarAP == 8 and nCSmax_local == 6:
            scaling = 4
        elif (NBarAP == 4 and nCSmax_local == 6) or (NBarAP == 8 and nCSmax_local == 12):
            scaling = 2
        else:
            scaling = 1

        nCSp = np.mod(nCS + nCSmax_local * np.floor((pBar - 1000) / scaling) / (NBarAP / scaling), nCSmax_local)
        fcsh, K = self._get_fcsh(nCSmax_local)
        alpha = (2 * np.pi / nCSmax_local) * (nCSp[:, np.newaxis] + fcsh[np.newaxis, :] / K)
        # For simplicity, return one alpha per port (taking first symbol’s value)
        return alpha[:, 0]
    
    def _get_wtdm(self):
        NSym = self.num_srs_symbols
        NPorts = self.num_srs_ports
        if self.enable_eight_port_tdm:
            wtdm = np.ones((NSym, NPorts))
            for s in range(1, NSym, 2):
                if NPorts >= 8:
                    wtdm[s, [2, 3, 6, 7]] = 0
                else:
                    wtdm[s, 1::2] = 0
            return wtdm
        else:
            return np.ones((NSym, NPorts))

    def _low_papr_sequence(self, u, v, alpha, Msc):
        seq = np.zeros((self.num_srs_ports, Msc), dtype=complex)
        for p in range(self.num_srs_ports):
            q = int(u) + p + 1
            zc = self._zc_sequence(Msc, q)
            seq[p, :] = zc * np.exp(1j * alpha[p])
        return seq

    # --- SRS Grid Generation ---
    @property
    def srs_grid(self):
        if not self.is_candidate_slot:
            return np.zeros((self.num_srs_ports, self.num_subcarriers, self.num_symbols_per_slot), dtype=complex)
        M_srs_all = self.m_srs_bsrs
        n_sub_srs = (M_srs_all * 12) // (self.k_tc * self.frequency_scaling_factor)
        Msc = n_sub_srs
        NSym = self.num_srs_symbols

        if self.group_seq_hopping.lower() == "grouphopping":
            u = self._group_number_hopping()
            v = np.zeros(NSym)
        elif self.group_seq_hopping.lower() == "sequencehopping":
            u = np.mod(self.n_srs_id, 30) * np.ones(NSym)
            v = self._sequence_number_hopping()
        else:
            u = np.mod(self.n_srs_id, 30) * np.ones(NSym)
            v = np.zeros(NSym)

        if self.k_tc == 2:
            nCSmax = 8
        elif self.k_tc == 4:
            nCSmax = 12
        elif self.k_tc == 8:
            nCSmax = 6
        else:
            nCSmax = 8

        alpha = self._compute_alpha(nCSmax)
        wtdm = self._get_wtdm()

        A = np.zeros((self.num_srs_ports, self.num_subcarriers, self.num_symbols_per_slot), dtype=complex)
        for s in range(NSym):
            seq = self._low_papr_sequence(u[s], v[s], alpha, Msc)
            for p in range(self.num_srs_ports):
                seq[p, :] *= wtdm[s, p]
            l = self.symbol_start + s
            A[:, 0:Msc, l] = seq
        self._info = {
            "SeqGroup": u,
            "NSeq": v,
            "Alpha": alpha,  # one value per port (for simplicity)
            "SeqLength": Msc
        }
        return A

    @property
    def info(self):
        if hasattr(self, "_info"):
            return self._info
        else:
            _ = self.srs_grid
            return self._info

    def srs_mask(self):
        M_srs_all = self.m_srs_bsrs
        n_sub_srs = (M_srs_all * 12) // (self.k_tc * self.frequency_scaling_factor)
        mask = np.zeros((self.num_subcarriers, self.num_symbols_per_slot), dtype=bool)
        if self.is_candidate_slot:
            s0 = self.symbol_start
            s1 = s0 + self.num_srs_symbols
            mask[0:n_sub_srs, s0:s1] = True
        return mask

    def check_config(self):
        self.carrier.check_config()
        assert (self.symbol_start + self.num_srs_symbols) <= self.carrier.num_symbols_per_slot, \
            "Symbol allocation invalid for the given CP length."
        return True

    def show(self):
        print("=== SRS Configuration ===")
        super().show()
        self.carrier.show()
