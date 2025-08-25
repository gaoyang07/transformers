
<!--Copyright 2025 OpenMOSS and The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
http://www.apache.org/licenses/LICENSE-2.0
-->


# MOSS-TTSD

<div style="float: right;">
<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</div>
</div>

## Overview

MOSS-TTSD (text to spoken dialogue) is an open-source bilingual spoken dialogue synthesis model that supports both Chinese and English.

It can transform dialogue scripts between two speakers into natural, expressive conversational speech.

MOSS-TTSD supports voice cloning and long single-session speech generation, making it ideal for AI podcast production, interviews, and chats.

For detailed information about the model and demos, please refer to our [Blog-en](https://www.open-moss.com/en/moss-ttsd/) and [中文博客](https://www.open-moss.com/cn/moss-ttsd/). You can also find the model on [Hugging Face](https://huggingface.co/fnlp/MOSS-TTSD-v0.5) and try it out in the [Spaces demo](https://huggingface.co/spaces/fnlp/MOSS-TTSD).

## Highlights

- **Highly Expressive Dialogue Speech**: Built on unified semantic-acoustic neural audio codec, a pre-trained large language model, millions of hours of TTS data, and 400k hours synthetic and real conversational speech, MOSS-TTSD generates highly expressive, human-like dialogue speech with natural conversational prosody.
- **Two-Speaker Voice Cloning**: MOSS-TTSD supports zero-shot two speakers voice cloning and can generate conversational speech with accurate speaker swithcing based on dialogue scripts. Only 10 to 20 seconds of reference audio is needed.
- **Chinese-English Bilingual Support**: MOSS-TTSD enables highly expressive speech generation in both Chinese and English.
- **Long-Form Speech Generation**: Thanks to low-bitrate codec and training framework optimization, MOSS-TTSD has been trained for long speech generation (Training maximum length is 960s).
- **Fully Open Source & Commercial-Ready**: MOSS-TTSD and its future updates will be fully open-source and support free commercial use.


## Usage Tips

### Text Generation (No voice cloning)

For pure text-to-speech generation without voice cloning, using the model's default voice:

```python
import os
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.moss_ttsd.processing_moss_ttsd import MossTTSDProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = MossTTSDProcessor.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    audio_tokenizer_path="fnlp/XY_Tokenizer_TTSD_V0_hf"
)
tokenizer = AutoTokenizer.from_pretrained("fnlp/MOSS-TTSD-v0.5")
model = AutoModelForCausalLM.from_pretrained("fnlp/MOSS-TTSD-v0.5").to(device).eval()

# Conversation format
conversation = [
    {"role": "system", "content": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."},
    {"role": "user", "content": "人工智能浪潮正在席卷全球，给我们带来深刻变化"}
]

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audio = []

inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True).to(device)

outputs = model.generate(
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"], 
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    max_length=1024
)

prompt_token_length = processor.get_prompt_len(inputs.input_ids)
response = processor.decode(outputs[0], outputs[0], prompt_token_length=prompt_token_length)

if response.audio is not None:
    processor.save_audio(response.audio, "output.wav")
```

### Single Voice Cloning

For voice cloning with a single speaker reference audio:

```python
import os
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.moss_ttsd.processing_moss_ttsd import MossTTSDProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = MossTTSDProcessor.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    audio_tokenizer_path="fnlp/XY_Tokenizer_TTSD_V0_hf"
)
tokenizer = AutoTokenizer.from_pretrained("fnlp/MOSS-TTSD-v0.5")
model = AutoModelForCausalLM.from_pretrained("fnlp/MOSS-TTSD-v0.5").to(device).eval()

# Conversation format with reference audio
conversation = [
    {"role": "system", "content": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."},
    {"role": "assistant", "content": [{"type": "audio", "audio_path": "path/to/reference_audio.wav"}]},
    {"role": "user", "content": "人工智能浪潮正在席卷全球，给我们带来深刻变化"}
]

# Load reference audio
ref_audio, sr = torchaudio.load("path/to/reference_audio.wav")
if sr != processor.input_sample_rate:
    ref_audio = torchaudio.functional.resample(ref_audio, sr, processor.input_sample_rate)
if ref_audio.shape[0] > 1:
    ref_audio = ref_audio.mean(dim=0, keepdim=True)

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audio = [ref_audio.squeeze(0).numpy()]

inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True).to(device)

outputs = model.generate(
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"], 
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    max_length=1024
)

prompt_token_length = processor.get_prompt_len(inputs.input_ids)
response = processor.decode(outputs[0], outputs[0], prompt_token_length=prompt_token_length)

if response.audio is not None:
    processor.save_audio(response.audio, "output.wav")
```

### Dual Dialogue Voice Cloning

For generating dialogue with two different speaker voices using separate reference audios:

```python
import os
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.moss_ttsd.processing_moss_ttsd import MossTTSDProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = MossTTSDProcessor.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    audio_tokenizer_path="fnlp/XY_Tokenizer_TTSD_V0_hf"
)
tokenizer = AutoTokenizer.from_pretrained("fnlp/MOSS-TTSD-v0.5")
model = AutoModelForCausalLM.from_pretrained("fnlp/MOSS-TTSD-v0.5").to(device).eval()

# Conversation format with two reference audios
conversation = [
    {"role": "system", "content": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."},
    {"role": "assistant", "content": [
        {"type": "audio", "audio_path": "path/to/speaker1_reference.wav"},
        {"type": "audio", "audio_path": "path/to/speaker2_reference.wav"}
    ]},
    {"role": "user", "content": "[S1]你听说了吗，人工智能现在变得非常厉害！[S2]是啊，我听说现在TTS模型生成的声音已经非常逼真了"}
]

# Load both reference audios
ref_audio1, sr1 = torchaudio.load("path/to/speaker1_reference.wav")
if sr1 != processor.input_sample_rate:
    ref_audio1 = torchaudio.functional.resample(ref_audio1, sr1, processor.input_sample_rate)
if ref_audio1.shape[0] > 1:
    ref_audio1 = ref_audio1.mean(dim=0, keepdim=True)
    
ref_audio2, sr2 = torchaudio.load("path/to/speaker2_reference.wav")
if sr2 != processor.input_sample_rate:
    ref_audio2 = torchaudio.functional.resample(ref_audio2, sr2, processor.input_sample_rate)
if ref_audio2.shape[0] > 1:
    ref_audio2 = ref_audio2.mean(dim=0, keepdim=True)

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audio = [ref_audio1.squeeze(0).numpy(), ref_audio2.squeeze(0).numpy()]

inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True).to(device)

outputs = model.generate(
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"], 
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    max_length=1024
)

prompt_token_length = processor.get_prompt_len(inputs.input_ids)
response = processor.decode(outputs[0], outputs[0], prompt_token_length=prompt_token_length)

if response.audio is not None:
    processor.save_audio(response.audio, "output.wav")
```


## MossTTSDConfig

[[autodoc]] MossTTSDConfig

## MossTTSDModel

[[autodoc]] MossTTSDModel
    - forward

## MossTTSDForCausalLM

[[autodoc]] MossTTSDForCausalLM
    - forward

## MossTTSDProcessor

[[autodoc]] MossTTSDProcessor
    - __call__
    - from_pretrained
    - save_pretrained
