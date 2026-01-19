"""
Step 1: Sionna Tx-Rx pipeline example
Simulates a basic wireless link and saves Rx output for AI training.
"""
import numpy as np
import tensorflow as tf
from sionna.channel import AWGN
from sionna.modem import QAMModem

# Parameters
num_bits = 10000
mod_order = 4  # QPSK
snr_db = 10

# Generate random bits
tx_bits = np.random.randint(0, 2, num_bits)

# Modulate
modem = QAMModem(mod_order)
tx_symbols = modem.modulate(tx_bits)

# Channel (AWGN)
awgn = AWGN()
snr_linear = 10 ** (snr_db / 10)
rx_symbols = awgn(tf.constant(tx_symbols, dtype=tf.complex64), snr_linear)

# Demodulate (classical Rx)
rx_bits = modem.demodulate(rx_symbols)

# Save data for AI training
np.savez('rx_data.npz', tx_bits=tx_bits, rx_symbols=rx_symbols.numpy(), rx_bits=rx_bits.numpy())

print("Sionna Tx-Rx pipeline data generated and saved to rx_data.npz.")
