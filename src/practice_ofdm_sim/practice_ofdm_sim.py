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
    sers = []
    sers_theo = []
    constellation = np.array([
        complex(*QAM16_SYMBOLS[label]) for label in range(M)
    ])
    for snr_db in SNR:
        rng = np.random.default_rng()
        # Each row contains 100 QAM labels
        tx_labels = rng.integers(0, 16, size=(NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS))
        tx_symbols = constellation[tx_labels]
        # Apply an IFFT to each row, then transmit the rows consecutively.
        tx_time_symbols = fft.ifft(tx_symbols, axis=1)
        if not sers:
            plot_ofdm_spectrum(tx_symbols[0])
            plot_subcarrier_orthogonality(NUM_SUBCARRIERS)
        # Prepend each symbol's last CP_LEN samples before joining the stream.
        tx_with_cp = np.concatenate(
            (tx_time_symbols[:, -CP_LEN:], tx_time_symbols), axis=1
        )
        x_n = tx_with_cp.reshape(-1)

        # ------------- Transmission Model -------------  
        # h[j] is the complex gain of the path delayed by j samples.
        # For example, [1, 0, 0.4] adds a path delayed by two samples.
        h = [1, 0, 0.4]
        # h = [1]

        # Output of transmission is a **convolution** of Tx and the channel!
        channel_output = np.convolve(x_n, h, mode="full")

        # Channel also adds white noise (in an AWGN model)
        # Total complex noise power, find from SNR
        signal_power = np.mean(np.abs(channel_output) ** 2)
        noise_power = signal_power * (10 ** ( - snr_db / 10 ))
        w_n = np.sqrt(noise_power / 2) * (
            rng.normal(size=channel_output.size)
            + 1j * rng.normal(size=channel_output.size)
        )
        y_n = channel_output + w_n

        # Split into prefixed symbols, then discard each prefix before the FFT.
        rx_with_cp = y_n[:x_n.size].reshape(
            NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS + CP_LEN
        )
        rx_time_symbols = rx_with_cp[:, CP_LEN:]
        received_subcarriers = fft.fft(rx_time_symbols, axis=1)
        # Undo the known channel's amplitude and phase change on each subcarrier.
        H = fft.fft(h, n=NUM_SUBCARRIERS)
        received_subcarriers = received_subcarriers / H
        rx_labels = ml_receiver(received_subcarriers, constellation)
        rx_symbols = constellation[rx_labels]
        symbol_errors = np.count_nonzero(rx_labels != tx_labels)
        total_symbols = tx_labels.size
        ser = symbol_errors / total_symbols
        sers.append(ser)

        # AWGN-only approximation for reference, to show effect of multipath
        sers_theo.append(4*(1 - 1/np.sqrt(M)) * qfunc(np.sqrt((3 * signal_power) / ((M - 1)* noise_power) )))

    plt.figure()
    plt.semilogy(SNR, sers, "o", label="Simulation")
    plt.semilogy(SNR, sers_theo, color="red", label="AWGN approximation")
    plt.legend()
    plt.xlabel("SNR (dB)")
    plt.ylabel("Symbol error rate")
    plt.grid(True, which="both")
    plt.show()
