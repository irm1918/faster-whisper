start
class Word:
    """
    A NamedTuple representing a word in the transcription. It includes the start and end times,
    the word itself, and the probability of the word.
    
    Attributes:
        start (float): The start time of the word in the audio.
        end (float): The end time of the word in the audio.
        word (str): The transcribed word.
        probability (float): The probability of the word.
    """

class Segment:
    """
    A NamedTuple representing a segment in the transcription. It includes the id, seek, start and
    end times, the transcribed text, tokens, temperature, average log probability, compression
    ratio, no speech probability, and words.
    
    Attributes:
        id (int): The id of the segment.
        seek (int): The seek of the segment.
        start (float): The start time of the segment in the audio.
        end (float): The end time of the segment in the audio.
        text (str): The transcribed text of the segment.
        tokens (List[int]): The tokens of the transcribed text.
        temperature (float): The temperature used for sampling.
        avg_logprob (float): The average log probability of the transcribed text.
        compression_ratio (float): The compression ratio of the transcribed text.
        no_speech_prob (float): The no speech probability of the segment.
        words (Optional[List[Word]]): The words in the segment.
    """

class TranscriptionOptions:
    """
    A NamedTuple representing the options for transcription. It includes beam size, best of,
    patience, length penalty, repetition penalty, log probability threshold, no speech threshold,
    compression ratio threshold, condition on previous text, prompt reset on temperature,
    temperatures, initial prompt, prefix, suppress blank, suppress tokens, without timestamps,
    max initial timestamp, word timestamps, prepend punctuations, and append punctuations.
    
    Attributes:
        beam_size (int): Beam size to use for decoding.
        best_of (int): Number of candidates when sampling with non-zero temperature.
        patience (float): Beam search patience factor.
        length_penalty (float): Exponential length penalty constant.
        repetition_penalty (float): Penalty applied to the score of previously generated tokens.
        log_prob_threshold (Optional[float]): If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold (Optional[float]): If the no_speech probability is higher than this
            value AND the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compression_ratio_threshold (Optional[float]): If the gzip compression ratio is above this
            value, treat as failed.
        condition_on_previous_text (bool): If True, the previous output of the model is provided
            as a prompt for the next window.
        prompt_reset_on_temperature (float): Resets prompt if temperature is above this value.
        temperatures (List[float]): Temperature for sampling.
        initial_prompt (Optional[Union[str, Iterable[int]]]): Optional text string or iterable of
            token ids to provide as a prompt for the first window.
        prefix (Optional[str]): Optional text to provide as a prefix for the first window.
        suppress_blank (bool): Suppress blank outputs at the beginning of the sampling.
        suppress_tokens (Optional[List[int]]): List of token IDs to suppress.
        without_timestamps (bool): Only sample text tokens.
        max_initial_timestamp (float): The initial timestamp cannot be later than this.
        word_timestamps (bool): Extract word-level timestamps using the cross-attention pattern.
        prepend_punctuations (str): Merge these punctuation symbols with the next word.
        append_punctuations (str): Merge these punctuation symbols with the previous word.
    """

class TranscriptionInfo:
    """
    A NamedTuple representing the information of the transcription. It includes language,
    language probability, duration, all language probabilities, transcription options, and
    voice activity detection options.
    
    Attributes:
        language (str): The language spoken in the audio.
        language_probability (float): The probability of the detected language.
        duration (float): The duration of the audio.
        all_language_probs (Optional[List[Tuple[str, float]]]): The probabilities of all detected
            languages.
        transcription_options (TranscriptionOptions): The options used for the transcription.
        vad_options (VadOptions): The options used for the voice activity detection.
    """
end
start
class WhisperModel:
    def __init__(self, model_size_or_path: str, device: str='auto', device_index: Union[(int, List[int])]=0, compute_type: str='default', cpu_threads: int=0, num_workers: int=1, download_root: Optional[str]=None, local_files_only: bool=False):
        """
        Initializes the Whisper model.
        
        Args:
          model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, medium, medium.en, large-v1, or large-v2), a path to a converted
            model directory, or a CTranslate2-converted Whisper model ID from the Hugging Face Hub.
            When a size or a model ID is configured, the converted model is downloaded
            from the Hugging Face Hub.
          device: Device to use for computation ("cpu", "cuda", "auto").
          device_index: Device ID to use.
            The model can also be loaded on multiple GPUs by passing a list of IDs
            (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel
            when transcribe() is called from multiple Python threads (see also num_workers).
          compute_type: Type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html.
          cpu_threads: Number of threads to use when running on CPU (4 by default).
            A non zero value overrides the OMP_NUM_THREADS environment variable.
          num_workers: When transcribe() is called from multiple Python threads,
            having multiple workers enables true parallelism when running the model
            (concurrent calls to self.model.generate() will run in parallel).
            This can improve the global throughput at the cost of increased memory usage.
          download_root: Directory where the models should be saved. If not set, the models
            are saved in the standard Hugging Face cache directory.
          local_files_only:  If True, avoid downloading the file and return the path to the
            local cached file if it exists.
        """
    def transcribe(self, audio: Union[(str, BinaryIO, np.ndarray)], language: Optional[str]=None, task: str='transcribe', beam_size: int=5, best_of: int=5, patience: float=1, length_penalty: float=1, repetition_penalty: float=1, temperature: Union[(float, List[float], Tuple[(float, ...)])]=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], compression_ratio_threshold: Optional[float]=2.4, log_prob_threshold: Optional[float]=(- 1.0), no_speech_threshold: Optional[float]=0.6, condition_on_previous_text: bool=True, prompt_reset_on_temperature: float=0.5, initial_prompt: Optional[Union[(str, Iterable[int])]]=None, prefix: Optional[str]=None, suppress_blank: bool=True, suppress_tokens: Optional[List[int]]=[(- 1)], without_timestamps: bool=False, max_initial_timestamp: float=1.0, word_timestamps: bool=False, prepend_punctuations: str='"\'“¿([{-', append_punctuations: str='"\'.。,，!！?？:：”)]}、', vad_filter: bool=False, vad_parameters: Optional[Union[(dict, VadOptions)]]=None):
        """
        Transcribes an input file.
        
        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform.
          language: The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.
          task: Task to execute (transcribe or translate).
          beam_size: Beam size to use for decoding.
          best_of: Number of candidates when sampling with non-zero temperature.
          patience: Beam search patience factor.
          length_penalty: Exponential length penalty constant.
          repetition_penalty: Penalty applied to the score of previously generated tokens
            (set > 1 to penalize).
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `log_prob_threshold`.
          compression_ratio_threshold: If the gzip compression ratio is above this value,
            treat as failed.
          log_prob_threshold: If the average log probability over sampled tokens is
            below this value, treat as failed.
          no_speech_threshold: If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
          condition_on_previous_text: If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.
          prompt_reset_on_temperature: Resets prompt if temperature is above this value.
            Arg has effect only if condition_on_previous_text is True.
          initial_prompt: Optional text string or iterable of token ids to provide as a
            prompt for the first window.
          prefix: Optional text to provide as a prefix for the first window.
          suppress_blank: Suppress blank outputs at the beginning of the sampling.
          suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
            of symbols as defined in the model config.json file.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.
          word_timestamps: Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.
          prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the next word
          append_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the previous word
          vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model
            https://github.com/snakers4/silero-vad.
          vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
            parameters and default values in the class `VadOptions`).
        
        Returns:
          A tuple with:
        
            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """
end
start
def generate_segments():
    """
    Generates segments for transcription.
    
    Args:
      features: The audio features.
      tokenizer: The tokenizer to use.
      options: The transcription options.
      encoder_output: The output of the encoder. If None, the encoder is run on the segment.
      
    Returns:
      An iterable over the transcribed segments.
    """

def encode():
    """
    Encodes the audio features.
    
    Args:
      features: The audio features.
      
    Returns:
      The encoder output.
    """
end
start
def generate_with_fallback():
    """
    Generates transcriptions with fallback options. This function attempts to generate transcriptions
    using different temperature settings. If the transcriptions do not meet the specified thresholds,
    it falls back to the next temperature setting. If all attempts fail, it selects the result with 
    the highest average log probability.
    
    Args:
        encoder_output: The output from the encoder.
        prompt: The prompt to be used for generating the transcription.
        tokenizer: The tokenizer to be used for tokenizing the transcription.
        options: The transcription options.
        
    Returns:
        A tuple containing the result of the transcription, the average log probability, the temperature
        used, and the compression ratio.
    """

def get_prompt():
    """
    Gets the prompt for the transcription. The prompt is a list of tokens that is used to initiate the
    transcription process. It can include tokens from previous transcriptions, a special sequence start
    token, a no timestamps token, and tokens from a specified prefix.
    
    Args:
        tokenizer: The tokenizer to be used for tokenizing the prompt.
        previous_tokens: The tokens from the previous transcription.
        without_timestamps: A flag indicating whether timestamps should be included in the prompt.
        prefix: An optional prefix to be included in the prompt.
        
    Returns:
        A list of tokens representing the prompt.
    """

def add_word_timestamps():
    """
    Adds word-level timestamps to the transcriptions. This function uses the alignment between the
    encoder output and the text tokens to calculate the start and end times for each word. It also
    applies a median filter to smooth the timestamps and merges punctuations with the adjacent words.
    
    Args:
        segments: The transcribed segments.
        tokenizer: The tokenizer to be used for tokenizing the segments.
        encoder_output: The output from the encoder.
        num_frames: The number of frames in the audio.
        prepend_punctuations: The punctuations to be merged with the next word.
        append_punctuations: The punctuations to be merged with the previous word.
        last_speech_timestamp: The timestamp of the last speech segment.
    """

def find_alignment():
    """
    Finds the alignment between the encoder output and the text tokens. This function uses the model's
    alignment function to calculate the alignment. It then splits the text tokens into words and
    calculates the start and end times for each word.
    
    Args:
        tokenizer: The tokenizer to be used for tokenizing the text.
        text_tokens: The tokens representing the text.
        encoder_output: The output from the encoder.
        num_frames: The number of frames in the audio.
        median_filter_width: The width of the median filter used for smoothing the timestamps.
        
    Returns:
        A list of dictionaries, each containing the word, its tokens, its start and end times, and its
        probability.
    """
end
start
def restore_speech_timestamps():
    """
    Restores the original speech timestamps for each segment and word.

    Args:
      segments: An iterable of Segment objects.
      speech_chunks: A list of dictionaries representing speech chunks.
      sampling_rate: The sampling rate of the audio.

    Returns:
      An iterable of Segment objects with restored timestamps.
    """

def get_ctranslate2_storage():
    """
    Converts a numpy array into a CTranslate2 StorageView object.

    Args:
      segment: A numpy array representing a segment of audio.

    Returns:
      A CTranslate2 StorageView object.
    """

def get_compression_ratio():
    """
    Calculates the compression ratio of a given text.

    Args:
      text: A string of text.

    Returns:
      A float representing the compression ratio of the text.
    """

def get_suppressed_tokens():
    """
    Returns a list of suppressed tokens.

    Args:
      tokenizer: A tokenizer object.
      suppress_tokens: A list of tokens to suppress.

    Returns:
      A sorted set of suppressed tokens.
    """

def merge_punctuations():
    """
    Merges punctuations with the preceding or following word in the alignment.

    Args:
      alignment: A list of dictionaries representing the alignment of words and tokens.
      prepended: A string of punctuations to prepend to the following word.
      appended: A string of punctuations to append to the preceding word.
    """
end
