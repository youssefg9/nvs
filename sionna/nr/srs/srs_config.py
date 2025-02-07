"""
srs_config.py

Definition of the SRS configuration object for uplink Sounding Reference Signals.
This minimal configuration class holds only the parameters needed for the current SRS
pilot generation implementation.
"""

class SRSConfig:
    r"""SRSConfig for uplink SRS generation.

    Parameters
    ----------
    NumSRSPorts : int, optional
        Number of antenna ports used for SRS. Default is 1.
    SymbolStart : int, optional
        First OFDM symbol index in a slot where SRS is transmitted. Default is 13.
    NumSRSSymbols : int, optional
        Number of consecutive OFDM symbols allocated to SRS. Default is 1.
    FrequencyStart : int, optional
        Starting frequency index (within effective subcarriers) for SRS. Default is 0.
    Msc : int, optional
        Number of subcarriers allocated to SRS per OFDM symbol. Default is 12.
    KTC : int, optional
        Transmission comb number. (Not actively used in this minimal implementation.) Default is 2.
    FrequencyScalingFactor : int, optional
        Scaling factor for partial frequency sounding. (Not actively used here.) Default is 1.
    CyclicShift : int, optional
        Cyclic shift applied to the SRS sequence (in number of subcarriers). Default is 0.
    NSRSID : int, optional
        SRS scrambling identity. Default is 0.
    """
    def __init__(self,
                 num_srs_ports=1,
                 symbol_start=13,
                 num_srs_symbols=1,
                 frequency_start=0,
                 msc=12,
                 ktc=2,
                 frequency_scaling_factor=1,
                 cyclic_shift=0,
                 nsrsid=0):
        self.num_srs_ports = num_srs_ports
        self.symbol_start = symbol_start
        self.num_srs_symbols = num_srs_symbols
        self.frequency_start = frequency_start
        self.msc = msc
        self.ktc = ktc
        self.frequency_scaling_factor = frequency_scaling_factor
        self.cyclic_shift = cyclic_shift
        self.nsrsid = nsrsid

        self._validate_params()

    def _validate_params(self):
        if not isinstance(self.num_srs_ports, int) or self.num_srs_ports <= 0:
            raise ValueError("NumSRSPorts must be a positive integer.")
        if not 0 <= self.symbol_start < 14:
            raise ValueError("SymbolStart must be between 0 and 13.")
        if self.num_srs_symbols not in [1, 2, 4, 8, 10, 12, 14]:
            raise ValueError("NumSRSSymbols must be one of [1, 2, 4, 8, 10, 12, 14].")
        if not isinstance(self.frequency_start, int) or self.frequency_start < 0:
            raise ValueError("FrequencyStart must be a non-negative integer.")
        if not isinstance(self.msc, int) or self.msc <= 0:
            raise ValueError("Msc must be a positive integer.")
