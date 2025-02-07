import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import the ResourceGrid from Sionna's ofdm module.
from sionna.ofdm import ResourceGrid
# Import our SRS modules.
from srs_config import SRSConfig
from srs_pilot_pattern import SRSPilotPattern

# Create a Sionna ResourceGrid (parameters should be consistent with your overall system)
resource_grid = ResourceGrid(
    num_ofdm_symbols=14,
    fft_size=64,
    subcarrier_spacing=15e3,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=16,
    num_guard_carriers=(0, 0),
    dc_null=False,
    pilot_pattern=None  # We will overlay our SRS pilot pattern.
)

# --- Single-User Example ---
srs_cfg_single = SRSConfig(
    NumSRSPorts=2,
    SymbolStart=10,
    NumSRSSymbols=2,
    FrequencyStart=5,
    Msc=12,
    CyclicShift=1,
    NSRSID=123
)

srs_pilot_single = SRSPilotPattern(srs_cfg_single, resource_grid)

# Visualize the resource grid with the SRS pilot mask.
fig1 = resource_grid.show(tx_ind=0, tx_stream_ind=0)
plt.title("Resource Grid with SRS Pilot Pattern (Single User)")
plt.show()

# --- Multiuser Example ---
# Create two different SRS configurations for a multiuser scenario.
srs_cfg_user1 = SRSConfig(
    NumSRSPorts=2,
    SymbolStart=9,
    NumSRSSymbols=2,
    FrequencyStart=4,
    Msc=12,
    CyclicShift=0,
    NSRSID=100
)
srs_cfg_user2 = SRSConfig(
    NumSRSPorts=2,
    SymbolStart=11,
    NumSRSSymbols=2,
    FrequencyStart=20,
    Msc=12,
    CyclicShift=2,
    NSRSID=200
)
# Supply a list of configurations.
srs_pilot_multi = SRSPilotPattern([srs_cfg_user1, srs_cfg_user2], resource_grid)

# For visualization, you might loop over users/streams (here we print the pilot mask shape)
print("Multiuser SRS pilot mask shape:", srs_pilot_multi.mask.shape)
print("Multiuser SRS pilots shape:", srs_pilot_multi.pilots.shape)
