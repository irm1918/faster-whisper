"""We use the PyAV library to decode the audio: https://github.com/PyAV-Org/PyAV

The advantage of PyAV is that it bundles the FFmpeg libraries so there is no additional
system dependencies. FFmpeg does not need to be installed on the system.

However, the API is quite low-level so we need to manipulate audio frames directly.
"""

import io
import itertools

from typing import BinaryIO, Union

import av
import numpy as np


def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
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
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(input_file, metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    audio = audio.astype(np.float32) / 32768.0

    if split_stereo:
        left_channel = audio[0::2]
        right_channel = audio[1::2]
        return left_channel, right_channel

    return audio


def _ignore_invalid_frames(frames):
    """
    Filters out invalid frames from the provided iterable of frames.
    
    Args:
        frames (iterable): An iterable of audio frames.
    
    Yields:
        frame: Valid audio frames.
    """
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


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
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    """
    Resamples the provided audio frames using the given resampler.
    
    Args:
        frames (iterable): An iterable of audio frames.
        resampler (av.audio.resampler.AudioResampler): An audio resampler.
    
    Yields:
        frame: Resampled audio frames.
    """
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)
