"""
srs_config.py

Definition of a minimal SRS configuration object for uplink Sounding Reference Signals.
This configuration holds only the parameters needed by our SRS pilot generation.
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
        Transmission comb number (not actively used in this minimal implementation). Default is 2.
    FrequencyScalingFactor : int, optional
        Scaling factor for partial frequency sounding (not actively used here). Default is 1.
    CyclicShift : int, optional
        Cyclic shift applied to the SRS sequence (in number of subcarriers). Default is 0.
    NSRSID : int, optional
        SRS scrambling identity. Default is 0.
    """
    def __init__(self,
                 NumSRSPorts=1,
                 SymbolStart=13,
                 NumSRSSymbols=1,
                 FrequencyStart=0,
                 Msc=12,
                 KTC=2,
                 FrequencyScalingFactor=1,
                 CyclicShift=0,
                 NSRSID=0):
        self.NumSRSPorts = NumSRSPorts
        self.SymbolStart = SymbolStart
        self.NumSRSSymbols = NumSRSSymbols
        self.FrequencyStart = FrequencyStart
        self.Msc = Msc
        self.KTC = KTC
        self.FrequencyScalingFactor = FrequencyScalingFactor
        self.CyclicShift = CyclicShift
        self.NSRSID = NSRSID

        self._validate_params()

    def _validate_params(self):
        if not isinstance(self.NumSRSPorts, int) or self.NumSRSPorts <= 0:
            raise ValueError("NumSRSPorts must be a positive integer.")
        if not (0 <= self.SymbolStart < 14):
            raise ValueError("SymbolStart must be between 0 and 13.")
        if self.NumSRSSymbols not in [1, 2, 4, 8, 10, 12, 14]:
            raise ValueError("NumSRSSymbols must be one of [1, 2, 4, 8, 10, 12, 14].")
        if not isinstance(self.FrequencyStart, int) or self.FrequencyStart < 0:
            raise ValueError("FrequencyStart must be a non-negative integer.")
        if not isinstance(self.Msc, int) or self.Msc <= 0:
            raise ValueError("Msc must be a positive integer.")
