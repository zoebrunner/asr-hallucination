import torchaudio
import torch
import os
import re
import logging
import pickle
from decimal import Decimal, ROUND_DOWN
from datasets import load_dataset
from typing import Dict, Set, Any
from collections import defaultdict
from hallucination_args import process_commandline
import pyroomacoustics as pra
import numpy as np


class Signal:
    """
    Retrieves and prepares audio signals from the TEDLIUM3 corpus of TED talks for further processing.

    This class handles the loading of either the default test split or a custom 'speaker overlap' split,
    which consists of held back training data. The 'speaker overlap' split is useful for investigating
    speaker-dependent hallucinations in ASR systems.
    """

    def __init__(self, log: bool, speakeroverlap: bool, signal_dir: str = 'LIUM/tedlium',
                 release: str = "release3",
                 speakeroverlap_file: str = "speakeroverlap_setup/so_segments.txt",
                 speaker: str = None,
                 speech_sample: Any = None):
        """
        Initialise Signal class.

        :param bool log:
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if log else logging.WARNING)
        self.signal_dir = signal_dir
        self.speaker = speaker
        self.speakeroverlap = speakeroverlap
        self.release = release
        self.speakeroverlap_file = speakeroverlap_file
        self.speech_sample = speech_sample
        self.so_segments = self._process_speakeroverlap_data()
        self.pickle_filename = "speakeroverlap_data.pkl"  # TODO: manage directories correctly

    def load_from_hf(self, split: str = 'test') -> Dict[str, Any]:
        """
        Loads speech samples from huggingface for further processing.
        Either loads default test data, or custom 'speaker overlap' data split
        for investigating speaker-dependent hallucinations.

        If self.speakeroverlap is True, this method will attempt to load or create a pickled version
        of the held back 'speakeroverlap' training data.

        :param str split: The dataset split to load; defaults to 'test'.
        :return dict[str, array] data_hf: A dictionary of speech samples.
        :raise ValueError: If there's an error loading the dataset from HuggingFace.
        """
        # loads 'speaker overlap' data; ONLY use this if custom training split was used
        if self.speakeroverlap:
            return self._load_speakeroverlap_data()
        else:
            return self._load_default_data(split)

    def _load_speakeroverlap_data(self) -> Dict[str, Any]:
        """
        Loads speaker overlap data from pickle or HuggingFace.
        """

        self.logger.info("Loading held back training data for processing.")

        try:
            return self._load_from_pickle()
        except FileNotFoundError:
            return self._load_and_pickle_from_hf()

    def _load_from_pickle(self) -> Dict[str, Any]:
        """
        Load speaker overlap data from pickle file.
        """
        self.logger.info(f"Loading pickled speaker overlap data: {self.pickle_filename}")
        with open(self.pickle_filename, 'rb') as hb_data_file:
            return pickle.load(hb_data_file)

    def _load_and_pickle_from_hf(self) -> Dict[str, Any]:
        """
        Load speaker overlap data from HuggingFace and pickle it for later use.
        """
        self.logger.info("Loading speaker overlap data from HF")

        try:
            all_data_hf = load_dataset(self.signal_dir, self.release, split="train", streaming=True,
                                       trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Error loading dataset from HuggingFace: {e}")

        data_hf = {self.rename_file(segment): segment
                   for segment in all_data_hf
                   if self.rename_file(segment in self.so_segments)}

        # Pickle the data
        with open(self.pickle_filename, 'wb') as hb_data_file:
            pickle.dump(data_hf, hb_data_file)
        self.logger.info(f"Speaker overlap data successfully pickled: {self.pickle_filename}")

        return data_hf

    def _load_default_data(self, split: str) -> Dict[str, Any]:
        """
        Load default test split data from TEDLIUM3.
        """
        try:
            data_hf = load_dataset(self.signal_dir, self.release, split=split, streaming=True, trust_remote_code=True)
            self.logger.info(f"Data successfully loaded from {split} split.")
            return data_hf
        except Exception as e:
            raise ValueError(f"Error loading dataset from HuggingFace: {e}")

    @staticmethod
    def rename_file(segment: Dict[str, Any]):
        """
        Generates standardised filename for segments.
        I.e., circumvents rounding discrepancies in start/end times

        This method parses the segment ID to extract timing information,
        then formats it into a consistent filename structure.
        Floating-point precision issues are handled by Decimal.

        :args segment (dict): dictionary containing segment info, including 'id' key.
        :return: formatted filename string
        :raise: ValueError: if segment ID does not match expected format
        """

        timings_re = re.match(r'(?P<id>^[a-zA-Z_\d\.]+)-(?P<start>[\d\.]+)-(?P<end>[\d\.]+)', segment['id'])
        timings = re.match(timings_re, segment['id'])

        if not timings:
            raise ValueError(f"Invalid segment ID format: {segment['id']}")

        segment_id, start, end = timings.group('id', 'start', 'end')

        start_formatted = self.format_time(start)
        end_formatted = self.format_time(end)
        new_filename = f"{segment_id}-{start_formatted:0>6}-{end_formatted:0>6}.wav"

        return new_filename

    @staticmethod
    def _format_time(time_str: str) -> str:
        """
        Convert string into suitable time format; rounds to 7 decimals.

        :param str time_str: Time string to format.
        :return str: formatted time string
        """
        time = Decimal(time_str)

        return f"{time.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN):.6f}".replace('.', '')

    def _process_speakeroverlap_data(self) -> Set[str]:
        """
        Loads 'speaker overlap' segment names from a file, for later comparison against the entire database.
        Stores segments in a set, adhering to TEDLIUM3's standard naming convention of segment 'id'.
        Example: 'DavidRockwell_2002-0049484-0050449.wav'

        :return set[str]: set of segment ids
        """
        with open(self.speakeroverlap_file, 'r') as segment_file:
            so_segments = set(f'{seg_id.strip()}.wav' for seg_id in segment_file)
            return so_segments
