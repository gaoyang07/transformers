# coding=utf-8
# Copyright 2025 OpenMOSS and HuggingFace Inc. teams. All rights reserved.
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
"""Speech processor class for MOSS-TTSD"""

import math
import os
import re
from typing import Any, Optional, Union

import numpy as np

from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import is_torch_available, is_torchaudio_available


if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio


def process_jsonl_item(item: dict[str, Any]) -> dict[str, Any]:
    """Process JSONL data items and extract audio and text information according to the new format."""
    base_path = item.get("base_path", "")
    text = item.get("text", "")

    prompt_audio = None
    prompt_text = ""

    # Process prompt audio and text
    if "prompt_audio" in item and "prompt_text" in item:
        print("Using prompt_audio and prompt_text directly from item.")
        # If prompt_audio and prompt_text exist, use them directly
        prompt_audio_val = item["prompt_audio"]
        if prompt_audio_val:  # Only assign if not empty
            prompt_audio = prompt_audio_val
            prompt_text = item["prompt_text"]

            # Only perform path joining when prompt_audio is a string path
            if isinstance(prompt_audio, str) and base_path and prompt_audio:
                prompt_audio = os.path.join(base_path, prompt_audio)
    else:
        # Otherwise, merge speaker1 and speaker2 information
        prompt_audio_speaker1 = item.get("prompt_audio_speaker1", "")
        prompt_text_speaker1 = item.get("prompt_text_speaker1", "")
        prompt_audio_speaker2 = item.get("prompt_audio_speaker2", "")
        prompt_text_speaker2 = item.get("prompt_text_speaker2", "")

        has_speaker1_audio = (isinstance(prompt_audio_speaker1, str) and prompt_audio_speaker1) or isinstance(
            prompt_audio_speaker1, tuple
        )
        has_speaker2_audio = (isinstance(prompt_audio_speaker2, str) and prompt_audio_speaker2) or isinstance(
            prompt_audio_speaker2, tuple
        )

        if has_speaker1_audio or has_speaker2_audio:
            print("Using speaker1 and speaker2 information for prompt audio and text.")
            # Process audio: if it's a string path, perform path joining; if it's a tuple, use directly
            if isinstance(prompt_audio_speaker1, str):
                speaker1_audio = (
                    os.path.join(base_path, prompt_audio_speaker1)
                    if base_path and prompt_audio_speaker1
                    else prompt_audio_speaker1
                )
            else:
                speaker1_audio = prompt_audio_speaker1  # Use tuple directly

            if isinstance(prompt_audio_speaker2, str):
                speaker2_audio = (
                    os.path.join(base_path, prompt_audio_speaker2)
                    if base_path and prompt_audio_speaker2
                    else prompt_audio_speaker2
                )
            else:
                speaker2_audio = prompt_audio_speaker2  # Use tuple directly

            prompt_audio = {"speaker1": speaker1_audio, "speaker2": speaker2_audio}

        # Merge text
        temp_prompt_text = ""
        if prompt_text_speaker1:
            temp_prompt_text += f"[S1]{prompt_text_speaker1}"
        if prompt_text_speaker2:
            temp_prompt_text += f"[S2]{prompt_text_speaker2}"
        prompt_text = temp_prompt_text.strip()

    return {"text": text, "prompt_text": prompt_text, "prompt_audio": prompt_audio}


def process_audio_data(
    prompt_audio: Optional[Union[str, dict[str, Any], tuple[torch.Tensor, int]]], target_sample_rate: int = 16000
) -> Optional[torch.Tensor]:
    """Load audio data and return processed audio tensor."""
    if prompt_audio is None:
        return None
    try:
        if isinstance(prompt_audio, dict) and "speaker1" in prompt_audio and "speaker2" in prompt_audio:
            wav = merge_speaker_audios(prompt_audio["speaker1"], prompt_audio["speaker2"], target_sample_rate)
        else:
            wav, _ = load_audio_data(prompt_audio, target_sample_rate)
        return wav
    except Exception as e:
        print(f"Error loading audio data: {e}")
        raise


def merge_speaker_audios(
    wav1: Union[str, tuple[torch.Tensor, int]],
    wav2: Union[str, tuple[torch.Tensor, int]],
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """Merge audio data from two speakers."""
    try:
        wav1 = load_audio_data(wav1, target_sample_rate)
        wav2 = load_audio_data(wav2, target_sample_rate)
        return torch.cat([wav1, wav2], dim=1)
    except Exception as e:
        print(f"Error merging audio: {e}")
        raise


def load_audio_data(
    audio_input: Union[str, tuple[torch.Tensor, int]], target_sample_rate: int = 16000
) -> tuple[torch.Tensor, int]:
    """Load and resample audio data."""
    audio, sr = _load_single_audio(audio_input)
    return _resample_audio(audio, sr, target_sample_rate)


def _load_single_audio(audio_input: Union[str, tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, int]:
    """Load single audio file or return tuple directly."""
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        return audio_input
    elif isinstance(audio_input, str):
        return torchaudio.load(audio_input)
    else:
        raise ValueError(f"Unsupported audio input format: {type(audio_input)}")


def _resample_audio(
    audio_input: torch.Tensor, sampling_rate: int, target_sample_rate: int = 16000
) -> tuple[torch.Tensor, int]:
    """Resample audio to target sample rate and normalize channels."""
    if sampling_rate != target_sample_rate:
        audio_input = torchaudio.functional.resample(audio_input, sampling_rate, target_sample_rate)
    if audio_input.shape[0] > 1:
        audio_input = audio_input.mean(dim=0, keepdim=True)
    if len(audio_input.shape) == 1:
        audio_input = audio_input.unsqueeze(0)
    return audio_input, target_sample_rate


def shifting_inputs(
    input_ids: np.ndarray, pad_token_id: int, pad_token: int = 1024, max_channels: int = 8
) -> np.ndarray:
    """Shift input IDs for multi-channel processing."""
    seq_len = input_ids.shape[0]
    new_seq_len = seq_len + max_channels - 1
    shifted_input_ids = np.full((new_seq_len, max_channels), pad_token, dtype=np.int64)
    shifted_input_ids[:, 0] = np.full(new_seq_len, pad_token_id, dtype=np.int64)
    for i in range(max_channels):
        shifted_input_ids[i : (seq_len + i), i] = input_ids[:, i]
    return shifted_input_ids


def shifting_outputs(output_ids: torch.Tensor, speech_token_range: list[int], max_channels: int = 8) -> torch.Tensor:
    """Shift output IDs for multi-channel processing."""
    seq_len = output_ids.shape[1] - max_channels + 1
    speech_ids = torch.full((output_ids.shape[0], seq_len, max_channels), 0)
    for j in range(max_channels):
        speech_ids[..., j] = output_ids[:, j : seq_len + j, j]
        if j == 0:
            speech_ids[..., j] = speech_ids[..., j] - speech_token_range[0]
    return speech_ids


def rpadding(
    input_ids: list[np.ndarray], channels: int, pad_token_id: int, pad_token: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad input IDs for batch processing."""
    attention_masks = [np.ones(inputs.shape[0]) for inputs in input_ids]
    max_length = max(ids.shape[0] for ids in input_ids)
    padded_input_ids, padded_attns = [], []
    for ids, attn in zip(input_ids, attention_masks):
        pad_len = max_length - ids.shape[0]
        input_pad = np.full((pad_len, channels), pad_token)
        input_pad[:, 0] = pad_token_id
        padded_input_ids.append(np.concatenate([input_pad, ids]))
        attn_pad = np.zeros(pad_len)
        padded_attns.append(np.concatenate([attn_pad, attn]))
    input_ids = torch.tensor(np.stack(padded_input_ids))
    attention_mask = torch.tensor(np.stack(padded_attns))
    return input_ids, attention_mask


def find_max_valid_positions(data: torch.Tensor, invalid_value: int = 1024) -> list[list[torch.Tensor]]:
    """Find maximum valid positions in multi-channel tensor data."""
    mask = torch.all(data[:, :, 1:] != invalid_value, dim=2)

    valid_indices = torch.where(mask)

    result_tensors = [[] for _ in range(len(data))]
    if valid_indices[0].numel() == 0:
        return result_tensors

    grouped_indices = []
    current_group = []

    for i, seq_no in enumerate(valid_indices[0]):
        pos_id = valid_indices[1][i]
        if len(current_group) == 0 or seq_no > current_group[-1]:
            current_group.append(seq_no)
            grouped_indices.append([[pos_id, pos_id + 1]])
        elif pos_id == grouped_indices[-1][-1][-1]:
            grouped_indices[-1][-1][-1] += 1
        else:
            grouped_indices[-1].append([pos_id, pos_id + 1])

    for group_id, indices in zip(current_group, grouped_indices):
        for start_index, end_index in indices:
            result_tensors[group_id].append(data[group_id, start_index:end_index, :])

    return result_tensors


def normalize_text(text: str) -> str:
    """
    Normalize multi-speaker script.

    1. Don't preserve line breaks.
    2. Remove brackets for non-speaker tags (if [] doesn't contain S1/S2...Sx format, remove the brackets themselves).
    3. Remove decorative symbols: 【】《》（）『』「」"-“” .
    4. Internal punctuation ！；：、 → ，；only allow ？ and ，。
    5. Multiple 。 keep only the last one, others → ，。
    6. Replace consecutive "哈" (>=2) with "(笑)".
    7. Auto-recognize [S1] / [S2] … tags; if missing, treat as whole segment.
    """
    # Replace [1], [2] etc. format with [S1], [S2] etc. format
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)

    # Remove decorative characters
    remove_chars = '【】《》（）『』「」"-“”～~'

    # Remove brackets for non-speaker tags (keep content, only remove brackets themselves)
    text = re.sub(r"\[(?!S\d+\])([^\]]*)\]", r"\1", text)

    # Use positive lookahead to split text by speaker tags (tags themselves are still preserved)
    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    normalized_lines = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # Extract tags
        m = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = m.groups() if m else ("", seg)

        # Remove irrelevant symbols
        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)

        # Handle consecutive "哈" characters: replace 2 or more with "(笑)"
        content = re.sub(r"哈{2,}", "(笑)", content)

        # Handle English laughter (e.g., "haha", "ha ha")
        content = re.sub(r"\b(ha(\s*ha)+)\b", "(laughs)", content, flags=re.IGNORECASE)

        # First handle multi-character punctuation marks
        content = content.replace("——", "，")
        content = content.replace("……", "，")

        # Handle single-character internal punctuation marks
        internal_punct_map = str.maketrans(
            {"！": "，", "!": ",", "；": "，", ";": ",", "：": "，", ":": ",", "、": "，", "？": "，", "?": ","}
        )
        content = content.translate(internal_punct_map)
        content = content.strip()

        # Keep only the final period
        if len(content) > 1:
            last_ch = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
            body = content[:-1].replace("。", "，")
            content = body + last_ch

        normalized_lines.append(f"{tag}{content}".strip())

    return "".join(normalized_lines)


class MossTTSDProcessor(ProcessorMixin):
    r"""
    Constructs an MOSS-TTSD processor.

    This processor encapsulates a text tokenizer and an audio codec (XY_Tokenizer)
    to prepare inputs for the MossTTSDForConditionalGeneration model.

    Args:
        tokenizer (`AutoTokenizer`):
            The text tokenizer.
        feature_extractor (`AutoFeatureExtractor`):
            The feature extractor for audio processing.
        codec (`AutoModel`):
            The audio codec (MCplayer/XY_Tokenizer) used for encoding and decoding audio.
        chat_template (`str`, *optional*):
            The chat template to use for conversation formatting.
        speech_token_range (`List[int]`, *optional*, defaults to `[151665, 152689]`):
            The range of speech tokens.
        audio_bos_token (`str`, *optional*, defaults to `"<|begin_of_speech|>"`):
            The beginning of speech token.
        audio_eos_token (`str`, *optional*, defaults to `"<|end_of_speech|>"`):
            The end of speech token.
        audio_pad_token_id (`int`, *optional*, defaults to `1024`):
            The padding token ID for audio.
        **kwargs:
            Additional keyword arguments passed to the parent class.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        feature_extractor: Optional[AutoFeatureExtractor] = None,
        codec: Optional[AutoModel] = None,
        chat_template: Optional[str] = None,
        speech_token_range: Optional[list[int]] = None,
        audio_bos_token: str = "<|begin_of_speech|>",
        audio_eos_token: str = "<|end_of_speech|>",
        audio_pad_token_id: int = 1024,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.codec = codec
        self.feature_extractor = feature_extractor
        self.max_channels = codec.quantizer.num_quantizers if codec else 8
        self.input_sample_rate = codec.config.input_sample_rate if codec else 16000
        self.output_sample_rate = codec.config.output_sample_rate if codec else 16000
        self.encoder_downsample_rate = codec.config.encoder_downsample_rate if codec else 320
        self.chat_template = tokenizer.chat_template if hasattr(tokenizer, "chat_template") else chat_template
        self.speech_token_range = (
            tokenizer.speech_token_range
            if hasattr(tokenizer, "speech_token_range")
            else (speech_token_range or [151665, 152689])
        )
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        self.audio_pad_token_id = (
            tokenizer.audio_pad_token_id if hasattr(tokenizer, "audio_pad_token_id") else audio_pad_token_id
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        """Load all components from a pretrained model directory."""
        # Assume codec model is in the same repository or path, or specify codec_path
        codec_path = kwargs.pop("codec_path", os.path.join(pretrained_model_name_or_path, "XY_Tokenizer"))
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        assert isinstance(codec_path, str), f"Unsupported codec_path input format: {type(codec_path)}"

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        feature_extractor = AutoFeatureExtractor.from_pretrained(codec_path, trust_remote_code=True, **kwargs)
        codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True, **kwargs)

        return super().from_pretrained(
            pretrained_model_name_or_path,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            codec=codec,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get processor dictionary for loading."""
        processor_dict, kwargs = super().get_processor_dict(pretrained_model_name_or_path, **kwargs)
        processor_dict["codec"] = kwargs.pop("codec", None)
        if "speech_token_range" in kwargs:
            processor_dict["speech_token_range"] = kwargs.pop("speech_token_range")
        if "audio_bos_token" in kwargs:
            processor_dict["audio_bos_token"] = kwargs.pop("audio_bos_token")
        if "audio_eos_token" in kwargs:
            processor_dict["audio_eos_token"] = kwargs.pop("audio_eos_token")
        if "audio_pad_token_id" in kwargs:
            processor_dict["audio_pad_token_id"] = kwargs.pop("audio_pad_token_id")
        return processor_dict, kwargs

    def __call__(
        self,
        data: Union[dict[str, Any], list[dict[str, Any]]],
        batch_id: int = 0,
        use_normalize: bool = False,
        padding: Union[bool, str] = True,
        return_tensors: Optional[str] = "pt",
        silence_duration: float = 0.0,
        **kwargs,
    ) -> BatchEncoding:
        """
        Core processing method that converts user-friendly input to model input.
        """
        assert silence_duration >= 0
        if isinstance(data, dict):
            data = [data]

        input_ids_list = []
        actual_texts_data = []

        for i, item in enumerate(data):
            # 1. Prepare text
            system_prompt = item.get("system_prompt")
            processed_item = process_jsonl_item(item)
            text = processed_item["text"]
            prompt_text = processed_item["prompt_text"]

            # Merge text, if prompt_text is empty, full_text is just text
            full_text = prompt_text + text if prompt_text else text
            original_full_text = full_text  # Save original text

            # Apply text normalization based on parameter
            if use_normalize:
                full_text = normalize_text(full_text)

            # Replace speaker tags
            final_text = full_text.replace("[S1]", "<speaker1>").replace("[S2]", "<speaker2>")

            # Save actual text information used
            actual_texts_data.append(
                {
                    "index": batch_id + i,
                    "original_text": original_full_text,
                    "normalized_text": normalize_text(original_full_text) if use_normalize else None,
                    "final_text": final_text,
                    "use_normalize": use_normalize,
                }
            )

            # 2. Load audio
            audio_data = process_audio_data(processed_item["prompt_audio"])

            # 3. Convert to multi-channel input
            inputs = self.process_inputs_for_processor(
                final_text, audio_data, system_prompt, silence_duration, **kwargs
            )
            inputs = shifting_inputs(inputs, self.tokenizer.pad_token_id, max_channels=self.max_channels)
            input_ids_list.append(inputs)

        # 4. Batch padding
        if padding:
            input_ids, attention_mask = rpadding(input_ids_list, self.max_channels, self.tokenizer.pad_token_id)
        else:
            raise NotImplementedError("Unpadded batches are not supported yet.")

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        if return_tensors == "pt":
            return BatchEncoding(batch, tensor_type="pt")

        return BatchEncoding(batch)

    def process_inputs_for_processor(
        self,
        text: str,
        audio_data: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        silence_duration: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """Process inputs for the processor."""
        assert isinstance(text, str), "Data format is wrong."
        if isinstance(system_prompt, str):
            kwargs["system_prompt"] = system_prompt
        prompt = self.apply_chat_template(conversation=None, text=text, **kwargs)
        inputs1 = np.array(self.tokenizer.encode(prompt))
        input_ids = np.full((inputs1.shape[0], self.max_channels), self.audio_pad_token_id)
        input_ids[:, 0] = inputs1

        if audio_data is not None:
            try:
                silence_samples = int(silence_duration * self.input_sample_rate)
                silence = torch.zeros(audio_data.shape[0], silence_samples)
                wav = torch.cat([audio_data, silence], dim=1)
                feature = self.feature_extractor(
                    wav, sampling_rate=self.input_sample_rate, return_attention_mask=True, return_tensors="pt"
                )

                with torch.no_grad():
                    encode_result = self.codec.encode(feature)
                    audio_token = encode_result["audio_codes"][:, 0].permute(1, 0).cpu().numpy()
                audio_token[:, 0] = audio_token[:, 0] + self.speech_token_range[0]
                input_ids = np.concatenate([input_ids, audio_token])
                silence_tokens = silence_duration * self.input_sample_rate / self.encoder_downsample_rate
                silence_position = math.floor(silence_tokens / 10) * 10
                if silence_position > 0:
                    input_ids = input_ids[:-silence_position]
            except Exception as e:
                print(f"Error processing audio data: {e}")
                raise
        return input_ids

    def batch_decode(self, token_ids: torch.Tensor, *args, **kwargs) -> tuple[list[str], list[list[torch.Tensor]]]:
        """
        Decode token IDs to text and audio fragments.

        Args:
            token_ids (`torch.Tensor`):
                The token IDs to decode.
            *args, **kwargs:
                Additional arguments passed to the tokenizer's `batch_decode`.

        Returns:
            `Tuple[List[str], List[List[torch.Tensor]]]`: A tuple containing decoded text and audio fragments.
        """
        assert token_ids.ndim == 3 and token_ids.shape[2] == self.max_channels
        B, T, C = token_ids.shape
        text = self.tokenizer.batch_decode(token_ids[:, :, 0], *args, **kwargs)
        normal_token_ids = shifting_outputs(token_ids, self.speech_token_range, self.max_channels)
        audio_fragments = find_max_valid_positions(normal_token_ids, self.audio_pad_token_id)
        decode_audio = []
        for sequence_fragments in audio_fragments:
            if len(sequence_fragments):
                decode_audio.append(
                    self.codec.decode(
                        torch.cat([fragment.permute(1, 0).unsqueeze(dim=1) for fragment in sequence_fragments], dim=1),
                        overlap_seconds=10,
                    )["audio_values"]
                )
            else:
                decode_audio.append([])
        return text, decode_audio

    def decode(self, token_ids: torch.Tensor, *args, **kwargs) -> tuple[str, torch.Tensor]:
        """
        Decode token IDs to text and audio.

        Args:
            token_ids (`torch.Tensor`):
                The token IDs to decode.
            *args, **kwargs:
                Additional arguments passed to the tokenizer's `decode`.

        Returns:
            `Tuple[str, torch.Tensor]`: A tuple containing decoded text and audio.
        """
        assert token_ids.ndim == 2 and token_ids.shape[1] == self.max_channels
        T, C = token_ids.shape
        text = self.tokenizer.decode(token_ids[:, 0].squeeze(dim=-1), *args, **kwargs)
        normal_token_ids = shifting_outputs(token_ids.unsqueeze(dim=0), self.speech_token_range, self.max_channels)
        audio_fragments = find_max_valid_positions(normal_token_ids, self.audio_pad_token_id)[0]
        audio_fragments = torch.cat([fragment.permute(1, 0).unsqueeze(dim=1) for fragment in audio_fragments], dim=1)
        decode_audio = self.codec.decode(audio_fragments, overlap_seconds=10)["audio_values"]

        return text, decode_audio


__all__ = ["MossTTSDProcessor"]
