class FeatureExtractor():
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

    def fram_wave(self, waveform, center=True):
        """
        Transform a raw waveform into a list of smaller waveforms.

        Args:
            waveform (ndarray): The input waveform.
            center (bool, optional): If True, pads the input waveform by reflecting around the edge. Default is True.

        Returns:
            ndarray: A 2d array of frames. Each row is a frame.
        """

    def stft(self, frames, window):
        """
        Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal.

        Args:
            frames (ndarray): The input frames.
            window (ndarray): The window function.

        Returns:
            ndarray: A 2d array of complex numbers representing the STFT of each frame.
        """

    def __call__(self, waveform, padding=True):
        """
        Compute the log-Mel spectrogram of the provided audio.

        Args:
            waveform (ndarray): The input waveform.
            padding (bool, optional): If True, pads the input waveform to match the number of samples. Default is True.

        Returns:
            ndarray: A 2d array representing the log-Mel spectrogram.
        """