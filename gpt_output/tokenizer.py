class Tokenizer():
    """
    A simple wrapper around a tokenizers.Tokenizer.

    Attributes:
        tokenizer (tokenizers.Tokenizer): The tokenizer to be used.
        multilingual (bool): Indicates if the tokenizer is multilingual.
        task (str, optional): The task for which the tokenizer is used.
        language (str, optional): The language code for the tokenizer.

    Raises:
        ValueError: If the provided task or language code is not valid.
    """

    def __init__():
        """
        Initializes the Tokenizer class with the given tokenizer, task, and language.
        """

    def transcribe():
        """
        Returns the token id for the 'transcribe' token.
        """

    def translate():
        """
        Returns the token id for the 'translate' token.
        """

    def sot():
        """
        Returns the token id for the 'startoftranscript' token.
        """

    def sot_lm():
        """
        Returns the token id for the 'startoflm' token.
        """

    def sot_prev():
        """
        Returns the token id for the 'startofprev' token.
        """

    def eot():
        """
        Returns the token id for the 'endoftext' token.
        """

    def no_timestamps():
        """
        Returns the token id for the 'notimestamps' token.
        """

    def timestamp_begin():
        """
        Returns the token id for the beginning of a timestamp.
        """

    def sot_sequence():
        """
        Returns a sequence of start of transcript, language, and task tokens.
        """

    def encode():
        """
        Encodes the given text without adding special tokens and returns the ids.
        """

    def decode():
        """
        Decodes the given tokens into text, ignoring tokens after the 'endoftext' token.
        """

    def decode_with_timestamps():
        """
        Decodes the given tokens into text with timestamps.
        """

    def split_to_word_tokens():
        """
        Splits the given tokens into word tokens based on the language code.
        """

    def split_tokens_on_unicode():
        """
        Splits the given tokens into word tokens at any position where the tokens are decoded as valid unicode points.
        """

    def split_tokens_on_spaces():
        """
        Splits the given tokens into word tokens at spaces.
        """
