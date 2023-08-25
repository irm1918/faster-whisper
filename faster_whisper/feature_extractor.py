import numpy as np


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py  # noqa: E501
class FeatureExtractor:
    """
    A class used to extract features from audio signals. It implements methods for Short-Time Fourier 
    Transform (STFT) and Mel-frequency cepstral coefficients (MFCCs) calculation.
    
    Attributes:
        n_fft (int): The window size for the FFT. Default is 400.
        hop_length (int): The hop (or stride) size. Default is 160.
        chunk_length (int): The length of audio chunk. Default is 30.
        n_samples (int): The number of samples in the chunk.
        nb_max_frames (int): The maximum number of frames in the chunk.
        time_per_frame (float): The time duration of each frame.
        sampling_rate (int): The sampling rate of the audio. Default is 16000 Hz.
        mel_filters (ndarray): The Mel filter bank.
        
    """
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.time_per_frame = hop_length / sampling_rate
        self.sampling_rate = sampling_rate
        self.mel_filters = self.get_mel_filters(
            sampling_rate, n_fft, n_mels=feature_size
        )

    def get_mel_filters(self, sr, n_fft, n_mels=128, dtype=np.float32):
        """
        Compute a Mel filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size (n_mels, 1 + n_fft // 2).
        
        Args:
            sr (int): The sampling rate of the audio.
            n_fft (int): The FFT size.
            n_mels (int, optional): The number of Mel bands. Default is 128.
            dtype (type, optional): The type of the output array. Default is np.float32.
        
        Returns:
            ndarray: A 2d array of shape (n_mels, 1 + n_fft // 2) storing filterbank. Each row holds 1 filter.
        """
        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, n_mels + 2)

        mels = np.asanyarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

        return weights

    def fram_wave(self, waveform, center=True):
        """
        Transform a raw waveform into a list of smaller waveforms.
        
        Args:
            waveform (ndarray): The input waveform.
            center (bool, optional): If True, pads the input waveform by reflecting around the edge. Default is True.
        
        Returns:
            ndarray: A 2d array of frames. Each row is a frame.
        """
        frames = []
        for i in range(0, waveform.shape[0] + 1, self.hop_length):
            half_window = (self.n_fft - 1) // 2 + 1
            if center:
                start = i - half_window if i > half_window else 0
                end = (
                    i + half_window
                    if i < waveform.shape[0] - half_window
                    else waveform.shape[0]
                )

                frame = waveform[start:end]

                if start == 0:
                    padd_width = (-i + half_window, 0)
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

                elif end == waveform.shape[0]:
                    padd_width = (0, (i - waveform.shape[0] + half_window))
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

            else:
                frame = waveform[i : i + self.n_fft]
                frame_width = frame.shape[0]
                if frame_width < waveform.shape[0]:
                    frame = np.lib.pad(
                        frame,
                        pad_width=(0, self.n_fft - frame_width),
                        mode="constant",
                        constant_values=0,
                    )

            frames.append(frame)
        return np.stack(frames, 0)

    def stft(self, frames, window):
        """
        Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal.
        
        Args:
            frames (ndarray): The input frames.
            window (ndarray): The window function.
        
        Returns:
            ndarray: A 2d array of complex numbers representing the STFT of each frame.
        """
        frame_size = frames.shape[1]
        fft_size = self.n_fft

        if fft_size is None:
            fft_size = frame_size

        if fft_size < frame_size:
            raise ValueError("FFT size must greater or equal the frame size")
        # number of FFT bins to store
        num_fft_bins = (fft_size >> 1) + 1

        data = np.empty((len(frames), num_fft_bins), dtype=np.complex64)
        fft_signal = np.zeros(fft_size)

        for f, frame in enumerate(frames):
            if window is not None:
                np.multiply(frame, window, out=fft_signal[:frame_size])
            else:
                fft_signal[:frame_size] = frame
            data[f] = np.fft.fft(fft_signal, axis=0)[:num_fft_bins]
        return data.T

    def __call__(self, waveform, padding=True):
        """
        Compute the log-Mel spectrogram of the provided audio.
        
        Args:
            waveform (ndarray): The input waveform.
            padding (bool, optional): If True, pads the input waveform to match the number of samples. Default is True.
        
        Returns:
            ndarray: A 2d array representing the log-Mel spectrogram.
        """
        if padding:
            waveform = np.pad(waveform, [(0, self.n_samples)])

        window = np.hanning(self.n_fft + 1)[:-1]

        frames = self.fram_wave(waveform)
        stft = self.stft(frames, window=window)
        magnitudes = np.abs(stft[:, :-1]) ** 2

        filters = self.mel_filters
        mel_spec = filters @ magnitudes

        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec
