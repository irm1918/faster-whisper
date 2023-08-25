import string

from functools import cached_property
from typing import List, Optional, Tuple

import tokenizers


class Tokenizer:
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

    def __init__(
        self,
        tokenizer: tokenizers.Tokenizer,
        multilingual: bool,
        task: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """
        Initializes the Tokenizer class with the given tokenizer, task, and language.
        """
        self.tokenizer = tokenizer

        if multilingual:
            self.task = self.tokenizer.token_to_id("<|%s|>" % task)
            if self.task is None:
                raise ValueError("%s is not a valid task" % task)

            self.language_code = language
            self.language = self.tokenizer.token_to_id("<|%s|>" % language)
            if self.language is None:
                raise ValueError("%s is not a valid language code" % language)

        else:
            self.task = None
            self.language = None
            self.language_code = "en"

    @cached_property
    def transcribe(self) -> int:
        """
        Returns the token id for the 'transcribe' token.
        """
        return self.tokenizer.token_to_id("<|transcribe|>")

    @cached_property
    def translate(self) -> int:
        """
        Returns the token id for the 'translate' token.
        """
        return self.tokenizer.token_to_id("<|translate|>")

    @cached_property
    def sot(self) -> int:
        """
        Returns the token id for the 'startoftranscript' token.
        """
        return self.tokenizer.token_to_id("<|startoftranscript|>")

    @cached_property
    def sot_lm(self) -> int:
        """
        Returns the token id for the 'startoflm' token.
        """
        return self.tokenizer.token_to_id("<|startoflm|>")

    @cached_property
    def sot_prev(self) -> int:
        """
        Returns the token id for the 'startofprev' token.
        """
        return self.tokenizer.token_to_id("<|startofprev|>")

    @cached_property
    def eot(self) -> int:
        """
        Returns the token id for the 'endoftext' token.
        """
        return self.tokenizer.token_to_id("<|endoftext|>")

    @cached_property
    def no_timestamps(self) -> int:
        """
        Returns the token id for the 'notimestamps' token.
        """
        return self.tokenizer.token_to_id("<|notimestamps|>")

    @property
    def timestamp_begin(self) -> int:
        """
        Returns the token id for the beginning of a timestamp.
        """
        return self.no_timestamps + 1

    @property
    def sot_sequence(self) -> List[int]:
        """
        Returns a sequence of start of transcript, language, and task tokens.
        """
        sequence = [self.sot]

        if self.language is not None:
            sequence.append(self.language)

        if self.task is not None:
            sequence.append(self.task)

        return sequence

    def encode(self, text: str) -> List[int]:
        """
        Encodes the given text without adding special tokens and returns the ids.
        """
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes the given tokens into text, ignoring tokens after the 'endoftext' token.
        """
        text_tokens = [token for token in tokens if token < self.eot]
        return self.tokenizer.decode(text_tokens)

    def decode_with_timestamps(self, tokens: List[int]) -> str:
        """
        Decodes the given tokens into text with timestamps.
        """
        outputs = [[]]

        for token in tokens:
            if token >= self.timestamp_begin:
                timestamp = f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)

        return "".join(
            [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        )

    def split_to_word_tokens(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Splits the given tokens into word tokens based on the language code.
        """
        if self.language_code in {"zh", "ja", "th", "lo", "my"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)

        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Splits the given tokens into word tokens at any position where the tokens are decoded as valid unicode points.
        """
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            try:
                replacement_char_index = decoded.index(replacement_char)
                replacement_char_index += unicode_offset
            except ValueError:
                replacement_char_index = None

            if replacement_char_index is None or (
                replacement_char_index < len(decoded_full)
                and decoded_full[replacement_char_index] == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Splits the given tokens into word tokens at spaces.
        """
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens
