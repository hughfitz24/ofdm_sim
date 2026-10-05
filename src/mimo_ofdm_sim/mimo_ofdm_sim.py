from scipy import fft, special
import numpy as np
import matplotlib.pyplot as plt

TOTAL_TX_SYMBOLS = 100_000
NUM_SUBCARRIERS = 20
CP_LEN = 5
NUM_OFDM_SYMBOLS = int(TOTAL_TX_SYMBOLS/NUM_SUBCARRIERS)
SNR = np.arange(0, 20.5, 0.5)

QAM16_SYMBOLS = {
    0: [-3/2, 3/2], 
    1: [-3/2, 1/2], 
    2: [-3/2, -1/2], 
    3: [-3/2, -3/2], 
    4: [-1/2, 3/2], 
    5: [-1/2, 1/2], 
    6: [-1/2, -1/2], 
    7: [-1/2, -3/2], 
    8: [1/2, 3/2], 
    9: [1/2, 1/2], 
    10: [1/2, -1/2], 
    11: [1/2, -3/2], 
    12: [3/2, 3/2], 
    13: [3/2, 1/2], 
    14: [3/2, -1/2], 
    15: [3/2, -3/2], 
}

M = 16 # 16-ary QAM

def ml_receiver(received, constellation):
    # ML receiver
    samples = np.asarray(received).reshape(-1)
    indices = np.empty(samples.size, dtype=np.intp)
    # Avoid allocating a full stream-by-constellation distance matrix.
    batch_size = 65536
    for start in range(0, samples.size, batch_size):
        batch = samples[start:start + batch_size]
        distances_squared = np.abs(batch[:, None] - constellation[None, :]) ** 2
        indices[start:start + batch_size] = np.argmin(distances_squared, axis=1)
    return indices.reshape(np.shape(received))

def qfunc(x):
    return 0.5-0.5*special.erf(x/np.sqrt(2))


def plot_ofdm_spectrum(subcarrier_symbols):
    """Plot one useful OFDM symbol (without CP) and its individual carriers."""
    num_carriers = len(subcarrier_symbols)
    samples = np.arange(num_carriers)
    carrier_indices = fft.fftfreq(num_carriers) * num_carriers
    # Each row is one carrier's contribution to the IFFT output.
    carriers = subcarrier_symbols[:, None] / num_carriers * np.exp(
        2j * np.pi * carrier_indices[:, None] * samples / num_carriers
    )
    # Zero padding samples the rectangular symbol's spectrum between FFT bins;
    # an ordinary N-point FFT only samples the peaks and the orthogonal nulls.
    n_fft = 128 * num_carriers
    frequency = fft.fftshift(fft.fftfreq(n_fft)) * num_carriers
    spectra = fft.fftshift(fft.fft(carriers, n=n_fft, axis=1), axes=1)
    carrier_power = np.abs(spectra) ** 2
    total_power = np.abs(spectra.sum(axis=0)) ** 2
    reference_power = carrier_power.max()

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for index in range(num_carriers):
        color = f"C{index % 10}"
        axes[0].plot(
            frequency,
            10 * np.log10(np.maximum(carrier_power[index] / reference_power, 1e-8)),
            color=color, alpha=0.45, linewidth=0.8,
        )
        axes[1].plot(
            frequency, carrier_power[index] / reference_power,
            color=color, linewidth=1,
            label=f"Carrier {int(round(carrier_indices[index]))}"
            if abs(carrier_indices[index]) <= 3 else None,
        )
    axes[0].plot(
        frequency, 10 * np.log10(np.maximum(total_power / reference_power, 1e-8)),
        color="black", linewidth=1, label="Combined OFDM symbol",
    )
    axes[0].set_ylim(-60, 15)
    axes[0].set_xlim(frequency[0], frequency[-1])
    axes[0].set_ylabel("Relative power (dB)")
    axes[0].set_title("One transmitted OFDM symbol: individual subcarriers and total")
    axes[0].legend()
    axes[1].set_xlim(-3.5, 3.5)
    axes[1].set_ylabel("Relative power (linear)")
    axes[1].set_title("Centre subcarriers: overlapping sidelobes")
    for centre in range(-3, 4):
        axes[1].axvline(centre, color="grey", linestyle=":", alpha=0.5)
    axes[1].set_xticks(range(-3, 4))
    axes[1].legend(ncol=4, fontsize="small")
    for axis in axes:
        axis.set_xlabel("Frequency / subcarrier spacing (baseband)")
        axis.grid(True)
    fig.tight_layout()


def plot_subcarrier_orthogonality(num_carriers):
    """Show normalized inner products over the useful OFDM symbol interval."""
    samples = np.arange(num_carriers)
    carrier_indices = np.arange(num_carriers)
    # Use unit-amplitude basis tones so the plot is independent of QAM data.
    basis = np.exp(
        2j * np.pi * carrier_indices[:, None] * samples / num_carriers
    )
    correlations = basis.conj() @ basis.T / num_carriers
    off_diagonal = correlations.copy()
    np.fill_diagonal(off_diagonal, 0)

    fig, axis = plt.subplots(figsize=(8, 7))
    heatmap = axis.imshow(
        np.abs(correlations), origin="lower", vmin=0, vmax=1,
        interpolation="nearest", cmap="viridis",
    )
    fig.colorbar(heatmap, ax=axis, label="Normalized inner-product magnitude")
    axis.set_xlabel("Subcarrier index l (IFFT bin)")
    axis.set_ylabel("Subcarrier index k (IFFT bin)")
    axis.set_title(
        "Subcarrier orthogonality over one useful symbol (without CP)\n"
        f"Diagonal = 1; largest off-diagonal = {np.abs(off_diagonal).max():.2e}"
    )
    fig.tight_layout()

def main() -> None:
    # 2x2 MIMO 
    num_tx = 2
    num_rx = 2
    rng = np.random.default_rng(42)

    constellation = np.array([
        complex(*QAM16_SYMBOLS[label]) for label in range(M)
    ])

    # Split the existing total transmit power across both antennas.
    tx_scale = 1 / np.sqrt(num_tx)

    # Shape: (transmit antenna, OFDM symbol, subcarrier).
    # TOTAL_TX_SYMBOLS now means QAM symbols PER transmit antenna.
    tx_labels = rng.integers(
        0, M,
        size=(num_tx, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS),
    )
    tx_symbols = tx_scale * constellation[tx_labels]

    # IFFT and CP insertion independently on each antenna.
    tx_time = fft.ifft(tx_symbols, axis=-1)
    tx_with_cp = np.concatenate(
        (tx_time[..., -CP_LEN:], tx_time), axis=-1
    )
    x_n = tx_with_cp.reshape(num_tx, -1)

    # h[receive antenna, transmit antenna, delay].
    # A fixed illustrative channel with three taps per antenna pair.
    h = np.array([
        [[1.0, 0, 0.4], [0.3 + 0.2j, 0, 0.1]],
        [[0.2 - 0.1j, 0, 0.15j], [1.0, 0, -0.3]],
    ], dtype=complex)

    if CP_LEN < h.shape[-1] - 1:
        raise ValueError("CP must cover the channel's maximum delay")

    # Each receive antenna gets the SUM of both transmitted streams,
    # with a different channel for each transmitter–receiver pair.
    output_len = x_n.shape[-1] + h.shape[-1] - 1
    channel_output = np.zeros((num_rx, output_len), dtype=complex)

    for r in range(num_rx):
        for t in range(num_tx):
            channel_output[r] += np.convolve(
                x_n[t], h[r, t], mode="full"
            )

    # Frequency response: (subcarrier, receive antenna, transmit antenna).
    H = fft.fft(h, n=NUM_SUBCARRIERS, axis=-1)
    H = np.moveaxis(H, -1, 0)

    # TODO: What is zero-forcing? 
    # Zero-forcing equalizer: one matrix pseudoinverse per subcarrier.
    # Shape: (subcarrier, transmit antenna, receive antenna).
    W = np.linalg.pinv(H)

    # Define SNR using average received time-domain power per antenna.
    signal_power = np.mean(
        np.abs(channel_output[:, :x_n.shape[-1]]) ** 2
    )

    sers = []
    for snr_db in SNR:
        noise_power = signal_power * 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (
            rng.normal(size=channel_output.shape)
            + 1j * rng.normal(size=channel_output.shape)
        )
        y_n = channel_output + noise

        # Remove CP and FFT independently on each receive antenna.
        rx_with_cp = y_n[:, :x_n.shape[-1]].reshape(
            num_rx, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS + CP_LEN
        )
        received = fft.fft(rx_with_cp[..., CP_LEN:], axis=-1)

        # For every OFDM symbol and subcarrier, calculate x_hat = W @ y.
        # received: (receive antenna, OFDM symbol, subcarrier)
        # estimated: (transmit antenna, OFDM symbol, subcarrier)
        estimated = np.einsum("ktr,rsk->tsk", W, received)

        # Restore the original constellation amplitude before slicing.
        rx_labels = ml_receiver(estimated / tx_scale, constellation)
        sers.append(np.mean(rx_labels != tx_labels))

    plt.figure()
    plt.semilogy(SNR, sers, "o-", label="2×2 MIMO OFDM, zero forcing")
    plt.xlabel("Received SNR (dB)")
    plt.ylabel("Symbol error rate")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
