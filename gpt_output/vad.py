class VadOptions():
    """
    A class used to represent the Voice Activity Detection (VAD) options.

    Attributes:
        threshold (float): Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
            probabilities above this value are considered as speech. Default value is 0.5.
        min_speech_duration_ms (int): Minimum duration of speech chunks in milliseconds. Chunks shorter than this 
            are discarded. Default value is 250.
        max_speech_duration_s (float): Maximum duration of speech chunks in seconds. Chunks longer than this will be 
            split. Default value is infinity.
        min_silence_duration_ms (int): Minimum duration of silence chunks in milliseconds. Chunks shorter than this 
            are discarded. Default value is 2000.
        window_size_samples (int): Size of audio chunks fed to the silero VAD model. Default value is 1024.
        speech_pad_ms (int): Padding added to each side of the final speech chunks in milliseconds. Default value is 400.
    """

def get_speech_timestamps():
    """
    This method is used for splitting long audios into speech chunks using silero VAD.

    Args:
        audio (np.ndarray): One dimensional float array representing the audio.
        vad_options (Optional[VadOptions]): Options for VAD processing. Default is None.
        kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
        List[dict]: List of dictionaries containing begin and end samples of each speech chunk.
    """

def collect_chunks():
    """
    Collects and concatenates audio chunks.

    Args:
        audio (np.ndarray): One dimensional float array representing the audio.
        chunks (List[dict]): List of dictionaries containing begin and end samples of each speech chunk.

    Returns:
        np.ndarray: Concatenated audio chunks.
    """

class SpeechTimestampsMap():
    """
    A helper class to restore original speech timestamps.

    Attributes:
        sampling_rate (int): The sampling rate of the audio.
        time_precision (int): The precision of the time in number of decimal places. Default value is 2.
    """

def get_vad_model():
    """
    Returns the VAD model instance.

    Returns:
        SileroVADModel: The VAD model instance.
    """

class SileroVADModel():
    """
    A class used to represent the Silero Voice Activity Detection (VAD) model.

    Attributes:
        path (str): The path to the Silero VAD model.
    """
