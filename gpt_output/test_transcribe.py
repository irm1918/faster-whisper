def test_transcribe():
    """
    Tests the transcribe function of the WhisperModel class with a given audio path.
    It checks if the language probabilities, duration, and segment text are as expected.
    """

def test_prefix_with_timestamps():
    """
    Tests the transcribe function of the WhisperModel class with a given prefix.
    It checks if the number of segments and the segment text are as expected.
    """

def test_vad():
    """
    Tests the transcribe function of the WhisperModel class with Voice Activity Detection (VAD) parameters.
    It checks if the number of segments, the segment text, and the VAD options are as expected.
    """

def test_stereo_diarization():
    """
    Tests the transcribe function of the WhisperModel class with a stereo audio.
    It checks if the transcriptions of the left and right channels are as expected.
    """