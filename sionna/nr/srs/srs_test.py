import matplotlib.pyplot as plt
from sionna.ofdm import ResourceGrid
from srs_config import SRSConfig
from srs_pilot_pattern import SRSPilotPattern
from sionna.nr import CarrierConfig

# Create a carrier configuration (adjust parameters as needed).
carrier = CarrierConfig(
    n_size_grid=4,              # for example: 4 resource blocks => 48 subcarriers
    num_symbols_per_slot=14,    # typical slot length
    slot_number=0,
    frame_number=0
)
carrier.check_config()

# Create a ResourceGrid using the carrier.
resource_grid = ResourceGrid(
    num_ofdm_symbols=carrier.num_symbols_per_slot,
    fft_size=carrier.n_size_grid * 12,
    subcarrier_spacing=15e3,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=16,
    num_guard_carriers=(0, 0),
    dc_null=False,
    pilot_pattern=None  # We will later combine pilot patterns.
)

# --- Single-User SRS Only Example ---
srs_cfg_single = SRSConfig(carrier_config=carrier,
                           num_srs_ports=2,
                           symbol_start=10,
                           num_srs_symbols=2,
                           srs_period="on",
                           c_srs=0,
                           b_srs=0,
                           k_tc=2,
                           group_seq_hopping="neither",
                           n_srs_id=123,
                           frequency_scaling_factor=1,
                           cyclic_shift=1,
                           cyclic_shift_hopping=False,
                           enable_eight_port_tdm=False)

srs_pattern_single = SRSPilotPattern(srs_configs=srs_cfg_single)
# Visualize the SRS pilot mask (for TX 0, port 0)
fig1 = resource_grid.show(tx_ind=0, tx_stream_ind=0)
plt.title("Resource Grid with SRS Pilot Pattern (Single User)")
plt.show()

# --- Multiuser Example with Merging ---
# Create two SRS configurations (for two different users/transmitters).
srs_cfg_user1 = SRSConfig(carrier_config=carrier,
                          num_srs_ports=2,
                          symbol_start=9,
                          num_srs_symbols=2,
                          srs_period="on",
                          c_srs=0,
                          b_srs=0,
                          k_tc=2,
                          group_seq_hopping="neither",
                          n_srs_id=100,
                          frequency_scaling_factor=1,
                          cyclic_shift=0,
                          cyclic_shift_hopping=False,
                          enable_eight_port_tdm=False)

srs_cfg_user2 = SRSConfig(carrier_config=carrier,
                          num_srs_ports=2,
                          symbol_start=11,
                          num_srs_symbols=2,
                          srs_period="on",
                          c_srs=0,
                          b_srs=0,
                          k_tc=2,
                          group_seq_hopping="neither",
                          n_srs_id=200,
                          frequency_scaling_factor=1,
                          cyclic_shift=2,
                          cyclic_shift_hopping=False,
                          enable_eight_port_tdm=False)

# Assume we already have a DMRS pilot pattern (for example, from PUSCH).
# For demonstration, we create a dummy base pattern with the same grid dimensions.
# In practice, this would be your actual PUSCH-DMRS pilot pattern.
dummy_mask = (resource_grid.pilot_pattern.mask.numpy() if resource_grid.pilot_pattern is not None
              else 0 * np.ones((1, 1, carrier.num_symbols_per_slot, carrier.n_size_grid * 12), dtype=bool))
dummy_pilots = (resource_grid.pilot_pattern.pilots.numpy() if resource_grid.pilot_pattern is not None
                else 0 * np.ones((1, 1, np.sum(dummy_mask[0,0])), dtype=complex))
# For the purpose of this example, we set the dummy base pattern to all False.
dummy_mask[:] = False
from sionna.ofdm.pilot_pattern import PilotPattern
base_pattern = PilotPattern(dummy_mask, dummy_pilots, trainable=False, normalize=False)

# Create a multiuser SRS pilot pattern and merge it with the base pattern.
srs_pattern_multi = SRSPilotPattern(srs_configs=[srs_cfg_user1, srs_cfg_user2],
                                    base_pattern=base_pattern)

print("Multiuser SRS pilot mask shape:", srs_pattern_multi.mask.shape)
print("Multiuser SRS pilots shape:", srs_pattern_multi.pilots.shape)
