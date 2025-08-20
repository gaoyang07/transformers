# coding=utf-8
# Copyright 2025 OpenMOSS and HuggingFace Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature extractor class for XY-Tokenizer
"""
import math
from functools import partial
from typing import List, Optional, Union
from collections import deque

import torch
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor
from transformers.audio_utils import mel_filter_bank
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


class ExtractorIterator:
    def __init__(
        self,
        data,
        batch_size=1,
        chunk_length=30,
        overlap_seconds=10,
        overlap_side="both",
        sampling_rate=16000,
        encode_func=None,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.overlap_seconds = overlap_seconds
        self.overlap_side = overlap_side
        self.sampling_rate = sampling_rate

        # duration_size is the effective audio length of each processing
        self.chunk_size = int(self.chunk_length * self.sampling_rate)
        self.overlap_size = int(self.overlap_seconds * self.sampling_rate)
        self.duration_size = self.chunk_size - self.overlap_size
        assert (
            (overlap_side == "right") or (self.overlap_size % 2 == 0)
        ), '`overlap_seconds` must be divisible by 2 when `overlap_side` is "both".'
        # Note: here we only process non-overlapping blocks, and overlap will be processed outside (if needed)
        # or more explicitly in the iterator. For simplicity, we assume that the blocks are based on duration_size

        assert callable(encode_func)
        self.encode_func = encode_func

    def __iter__(self):
        """
        Return a generator that handles all batch processing logic.
        """
        # Batch-related variables are now local variables to __iter__, very clear
        batch_num = 0

        # NOTE: The chunk size output by chunk_and_pad_view is duration_size
        wav_tensor = torch.zeros(self.batch_size, 1, self.chunk_size)
        input_lengths = deque(maxlen=self.batch_size)
        input_seq_no = torch.zeros(self.batch_size, dtype=torch.long)
        
        right_boundary = self.get_right_boundary()

        for i, sample in enumerate(self.data):
            sample_chunks, sample_lengths, sample_seq_no = self.chunk_and_pad_view(sample, i)

            processed_in_sample = 0
            while processed_in_sample < len(sample_chunks):
                space_in_batch = self.batch_size - batch_num
                chunks_to_add = min(
                    space_in_batch, len(sample_chunks) - processed_in_sample
                )

                # Define slice range
                start_idx_sample = processed_in_sample
                end_idx_sample = processed_in_sample + chunks_to_add
                start_idx_batch = batch_num
                end_idx_batch = batch_num + chunks_to_add

                # Fill data
                wav_tensor[start_idx_batch:end_idx_batch] = sample_chunks[
                    start_idx_sample:end_idx_sample
                ]
                input_lengths.extend(sample_lengths[start_idx_sample:end_idx_sample])
                input_seq_no[start_idx_batch:end_idx_batch] = sample_seq_no[
                    start_idx_sample:end_idx_sample
                ]

                # Update counter
                batch_num += chunks_to_add
                processed_in_sample += chunks_to_add

                # If the batch is full, yield a copy and reset
                if batch_num == self.batch_size:
                    list_x = []
                    for xi, (_, right) in enumerate(input_lengths):
                        if right == right_boundary and torch.any(
                            wav_tensor[xi, :, right:] != 0
                        ):
                            list_x.append(wav_tensor[xi].reshape(-1).cpu().numpy())
                        else:
                            list_x.append(
                                wav_tensor[xi, :, :right].reshape(-1).cpu().numpy()
                            )
                    yield BatchFeature(
                        {
                            **self.encode_func(list_x),
                            "input_lengths": input_lengths,
                            "chunk_seq_no": input_seq_no.clone(),
                        }
                    )

                    # Reset batch counter and Tensor content
                    batch_num = 0
                    wav_tensor.zero_()
                    input_lengths.clear()
                    input_seq_no.zero_()

        # After the loop, process the last incomplete batch
        if batch_num > 0:
            list_x = []
            for xi in range(batch_num):
                _, right = input_lengths[xi]
                if right == right_boundary and torch.any(
                    wav_tensor[xi, :, right:] != 0
                ):
                    list_x.append(wav_tensor[xi].reshape(-1).cpu().numpy())
                else:
                    list_x.append(
                        wav_tensor[xi, :, :right].reshape(-1).cpu().numpy()
                    )
            yield BatchFeature(
                {
                    **self.encode_func(list_x),
                    "input_lengths": input_lengths,
                    "chunk_seq_no": input_seq_no[:batch_num].clone(),
                }
            )

    def chunk_and_pad_view(self, tensor, seq_no):
        x = tensor[0:1, :].unsqueeze(0)
        
        stride = self.duration_size
        kernel = self.chunk_size
        B, C, L = x.shape
    
        num_chunks = max(0, math.ceil((L - kernel) / stride)) + 1
        target_len = (num_chunks - 1) * stride + kernel
        padding_size = max(0, target_len - L)
        x_padded = F.pad(x, (0, padding_size), "constant", 0)
        output_tensor = (
            x_padded.unfold(dimension=2, size=kernel, step=stride)
            .squeeze(0)
            .transpose(0, 1)
        )
        
        output_lengths = self.get_windows_boundaries(num_chunks, L)
        output_seq_no = torch.full((num_chunks,), seq_no, dtype=torch.long)
        return output_tensor, output_lengths, output_seq_no

    def get_left_boundary(self):
        if self.overlap_side == "right":
            return 0
        else:
            return int(self.overlap_size / 2)

    def get_right_boundary(self):
        if self.overlap_side == "right":
            return self.duration_size
        else:
            return self.chunk_size - int(self.overlap_size / 2)
            
    def get_windows_boundaries(self, num_chunks, seq_len):
        left_boundary = self.get_left_boundary()
        right_boundary = self.get_right_boundary()

        output_lengths = [(left_boundary, right_boundary) for _ in range(num_chunks)]
        output_lengths[0] = (0, output_lengths[0][1])
        output_lengths[-1] = (
            output_lengths[-1][0], seq_len - self.duration_size * (num_chunks-1)
        )
        return output_lengths


class XYTokenizerFeatureExtractor(WhisperFeatureExtractor):
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        n_samples=480000,
        nb_max_frames=3000,
        padding_side="right",
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=False,
        max_frequency=None,
        batch_size=8,
        overlap_side="both", 
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            chunk_length=chunk_length,
            n_fft=n_fft,
            padding_value=padding_value,
            dither=dither,
            return_attention_mask=return_attention_mask,
            n_samples=n_samples,
            nb_max_frames=nb_max_frames,
            padding_side=padding_side,
            **kwargs,
        )
        self.max_frequency = (
            max_frequency if max_frequency is not None else sampling_rate / 2
        )
        self.batch_size = batch_size
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=self.max_frequency,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.overlap_side = overlap_side

    def __call__(
        self,
        raw_speech: Union[torch.Tensor, List[torch.Tensor]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[str] = "cpu",
        return_token_timestamps: Optional[bool] = None,
        overlap_seconds: int = 10,
        **kwargs,
    ) -> ExtractorIterator:

        if not isinstance(raw_speech, list):
            raw_speech = [raw_speech]

        return ExtractorIterator(
            raw_speech,
            batch_size=self.batch_size if self.batch_size else len(raw_speech), 
            chunk_length=self.chunk_length,
            overlap_seconds=overlap_seconds,
            overlap_side=self.overlap_side,
            sampling_rate=self.sampling_rate,
            encode_func=partial(
                super().__call__,
                truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                padding=padding,
                max_length=max_length,
                sampling_rate=sampling_rate,
                do_normalize=do_normalize,
                device=device,
                return_token_timestamps=return_token_timestamps,
                **kwargs,
            ),
        )


__all__ = ["XYTokenizerFeatureExtractor"]
