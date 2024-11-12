import numpy as np
import scipy as sp
from scipy import signal
from scipy.signal import butter, filtfilt


def ricker(peak_freq, samples, dt):
    """
    retorna a wavelet de Ricker e sua FFT
    """
    # Array do tempo
    t = np.arange(samples) * (dt / 1000)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)

    # Cálculo da wavelet de Ricker
    pi2_f2_t2 = (np.pi**2) * (peak_freq**2) * (t**2)
    ricker = (1.0 - 2.0 * pi2_f2_t2) * np.exp(-pi2_f2_t2)

    # Cálculo da FFT
    freqs = np.fft.rfftfreq(t.shape[0], d=dt / 1000)
    fft = np.abs(np.fft.rfft(ricker))
    fft = fft / np.max(fft)

    return t, ricker, freqs, fft


def butter(freq_hi, freq_low, samples, dt):
    # Calcular array de tempo
    t = np.arange(samples) * (dt / 1000)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)

    # Criar sinal de impulso
    imp = signal.unit_impulse(t.shape[0], "mid")

    # Aplicar filtro Butterworth passa-alta
    fs = 1000 * (1 / dt)
    b, a = signal.butter(4, freq_hi, fs=fs)
    response_zp = signal.filtfilt(b, a, imp)

    # Aplicar filtro Butterworth passa-baixa
    low_b, low_a = signal.butter(2, freq_low, "hp", fs=fs)
    butter_wvlt = signal.filtfilt(low_b, low_a, response_zp)

    # Normalizar a wavelet
    butter_wvlt = butter_wvlt / np.max(butter_wvlt)

    # Calcular a FFT
    freqs = np.fft.rfftfreq(t.shape[0], d=dt / 1000)
    fft = np.abs(np.fft.rfft(butter_wvlt))
    fft = fft / np.max(fft)

    return t, butter_wvlt, freqs, fft
