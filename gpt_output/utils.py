def get_assets_path():
    """
    Returns the path to the assets directory.
    
    Returns:
        str: The path to the assets directory.
    """

def get_logger():
    """
    Returns the module logger.
    
    Returns:
        Logger: The logger of the module 'faster_whisper'.
    """

def download_model(
    size_or_id: str,
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Downloads a CTranslate2 Whisper model from the Hugging Face Hub.

    The model is downloaded from https://huggingface.co/guillaumekln.

    Args:
        size_or_id (str): Size of the model to download (tiny, tiny.en, base, base.en, small, 
                          small.en, medium, medium.en, large-v1, or large-v2), or a 
                          CTranslate2-converted model ID from the Hugging Face Hub 
                          (e.g. guillaumekln/faster-whisper-large-v2).
        output_dir (str, optional): Directory where the model should be saved. If not set, the 
                                     model is saved in the cache directory.
        local_files_only (bool, optional): If True, avoid downloading the file and return the 
                                            path to the local cached file if it exists.
        cache_dir (str, optional): Path to the folder where cached files are stored.

    Returns:
        str: The path to the downloaded model.

    Raises:
        ValueError: If the model size is invalid.
    """

def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    """
    Formats a timestamp in the format 'HH:MM:SS.sss'.
    
    Args:
        seconds (float): The timestamp in seconds.
        always_include_hours (bool, optional): If True, always include the hours in the output 
                                               even if it is zero. Defaults to False.
        decimal_marker (str, optional): The character to use as the decimal marker. 
                                        Defaults to '.'.

    Returns:
        str: The formatted timestamp.

    Raises:
        AssertionError: If the input timestamp is negative.
    """

class disabled_tqdm(tqdm):
    """
    A subclass of tqdm that disables the progress bar.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the disabled_tqdm class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)