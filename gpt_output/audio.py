def decode_audio(input_file: Union[str, BinaryIO], sampling_rate: int = 16000, split_stereo: bool = False):
    """
    Decodes the audio from the provided input file and resamples it to the specified sampling rate.
    If split_stereo is enabled, it returns separate left and right channels.

    Args:
        input_file (Union[str, BinaryIO]): Path to the input file or a file-like object.
        sampling_rate (int, optional): Resample the audio to this sample rate. Defaults to 16000.
        split_stereo (bool, optional): Return separate left and right channels. Defaults to False.

    Returns:
        np.array: A float32 Numpy array. If `split_stereo` is enabled, the function returns a 2-tuple with the
                  separated left and right channels.
    """

def _ignore_invalid_frames(frames):
    """
    Filters out invalid frames from the provided iterable of frames.

    Args:
        frames (iterable): An iterable of audio frames.

    Yields:
        frame: Valid audio frames.
    """

def _group_frames(frames, num_samples=None):
    """
    Groups audio frames into larger frames with the specified number of samples.

    Args:
        frames (iterable): An iterable of audio frames.
        num_samples (int, optional): The number of samples for each grouped frame. If not specified, all frames
                                      are grouped into one. 

    Yields:
        frame: Grouped audio frames.
    """

def _resample_frames(frames, resampler):
    """
    Resamples the provided audio frames using the given resampler.

    Args:
        frames (iterable): An iterable of audio frames.
        resampler (av.audio.resampler.AudioResampler): An audio resampler.

    Yields:
        frame: Resampled audio frames.
    """