
<!--Copyright 2025 The HuggingFace Team. All rights reserved.
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
<<<<<<< HEAD

It can transform dialogue scripts between two speakers into natural, expressive conversational speech.

MOSS-TTSD supports voice cloning and long single-session speech generation, making it ideal for AI podcast production, interviews, and chats.

For detailed information about the model and demos, please refer to our [Blog-en](https://www.open-moss.com/en/moss-ttsd/) and [中文博客](https://www.open-moss.com/cn/moss-ttsd/). You can also find the model on [Hugging Face](https://huggingface.co/fnlp/MOSS-TTSD-v0.5) and try it out in the [Spaces demo](https://huggingface.co/spaces/fnlp/MOSS-TTSD).

## Highlights

- **Highly Expressive Dialogue Speech**: Built on unified semantic-acoustic neural audio codec, a pre-trained large language model, millions of hours of TTS data, and 400k hours synthetic and real conversational speech, MOSS-TTSD generates highly expressive, human-like dialogue speech with natural conversational prosody.
- **Two-Speaker Voice Cloning**: MOSS-TTSD supports zero-shot two speakers voice cloning and can generate conversational speech with accurate speaker swithcing based on dialogue scripts. Only 10 to 20 seconds of reference audio is needed.
- **Chinese-English Bilingual Support**: MOSS-TTSD enables highly expressive speech generation in both Chinese and English.
- **Long-Form Speech Generation**: Thanks to low-bitrate codec and training framework optimization, MOSS-TTSD has been trained for long speech generation (Training maximum length is 960s).
- **Fully Open Source & Commercial-Ready**: MOSS-TTSD and its future updates will be fully open-source and support free commercial use.
=======
It can transform dialogue scripts between two speakers into natural, expressive conversational speech.
MOSS-TTSD supports voice cloning and long single-session speech generation, making it ideal for AI podcast production, interviews, and chats.
 For detailed information about the model and demos, please refer to our [Blog-en](https://www.open-moss.com/en/moss-ttsd/) and [中文博客](https://www.open-moss.com/cn/moss-ttsd/). You can also find the model on [Hugging Face](https://huggingface.co/fnlp/MOSS-TTSD-v0.5) and try it out in the [Spaces demo](https://huggingface.co/spaces/fnlp/MOSS-TTSD).

## Highlights
>>>>>>> ee7f38c7c4 (update mossttsd readme)

- **Highly Expressive Dialogue Speech**: Built on unified semantic-acoustic neural audio codec, a pre-trained large language model, millions of hours of TTS data, and 400k hours synthetic and real conversational speech, MOSS-TTSD generates highly expressive, human-like dialogue speech with natural conversational prosody.
- **Two-Speaker Voice Cloning**: MOSS-TTSD supports zero-shot two speakers voice cloning and can generate conversational speech with accurate speaker swithcing based on dialogue scripts. Only 10 to 20 seconds of reference audio is needed.
- **Chinese-English Bilingual Support**: MOSS-TTSD enables highly expressive speech generation in both Chinese and English.
- **Long-Form Speech Generation**: Thanks to low-bitrate codec and training framework optimization, MOSS-TTSD has been trained for long speech generation (Training maximum length is 960s).
- **Fully Open Source & Commercial-Ready**: MOSS-TTSD and its future updates will be fully open-source and support free commercial use.


## Usage Tips

### Generation with Text (No voice cloning, using the model's random timbre)

```python
import os
import torchaudio
from transformers import AutoModelForCausalLM
from transformers.models.moss_ttsd.processor_moss_ttsd import MossTTSDProcessor
processor = MossTTSDProcessor.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    codec_path="fnlp/XY_Tokenizer_TTSD_V0_hf",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    trust_remote_code=True
).eval()
data = [{
    "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
    "text": "[S1]Speaker 1 dialogue content[S2]Speaker 2 dialogue content[S1]..."
}]
# Try to use the ExtractorIterator as an iterator
print("Trying iterator approach...", flush=True)
inputs = processor(data, use_normalize=True)
token_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
text, audios = processor.batch_decode(token_ids)
if not os.path.exists("outputs/"):
    os.mkdir("outputs/")
for i, data in enumerate(audios):
    for j, fragment in enumerate(data):
        print(f"Saving audio_{i}_{j}.wav...", flush=True)
        torchaudio.save(f"outputs/audio_{i}_{j}.wav", fragment.cpu(), 24000)
```

### Generation with Text and Audio (Separate speaker audio references)

```python
import os
import torchaudio
from transformers import AutoModelForCausalLM
from transformers.models.moss_ttsd.processor_moss_ttsd import MossTTSDProcessor
processor = MossTTSDProcessor.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    codec_path="fnlp/XY_Tokenizer_TTSD_V0_hf",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    trust_remote_code=True
).eval()
data = [{
    "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
    "base_path": "/path/to/audio/files",
    "text": "[S1]Speaker 1 dialogue content[S2]Speaker 2 dialogue content[S1]...",
    "prompt_audio_speaker1": "path/to/speaker1_audio.wav",
    "prompt_text_speaker1": "Reference text for speaker 1 voice cloning",
    "prompt_audio_speaker2": "path/to/speaker2_audio.wav", 
    "prompt_text_speaker2": "Reference text for speaker 2 voice cloning"
}]
# Try to use the ExtractorIterator as an iterator
print("Trying iterator approach...", flush=True)
inputs = processor(data, use_normalize=True)
token_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
text, audios = processor.batch_decode(token_ids)
if not os.path.exists("outputs/"):
    os.mkdir("outputs/")
for i, data in enumerate(audios):
    for j, fragment in enumerate(data):
        print(f"Saving audio_{i}_{j}.wav...", flush=True)
        torchaudio.save(f"outputs/audio_{i}_{j}.wav", fragment.cpu(), 24000)
```


### Generation with Text and Audio (Shared audio reference)

```python
import os
import torchaudio
from transformers import AutoModelForCausalLM
from transformers.models.moss_ttsd.processor_moss_ttsd import MossTTSDProcessor
processor = MossTTSDProcessor.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    codec_path="fnlp/XY_Tokenizer_TTSD_V0_hf",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "fnlp/MOSS-TTSD-v0.5",
    trust_remote_code=True
).eval()
data = [{
    "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
    "base_path": "/path/to/audio/files",
    "text": "[S1]Speaker 1 dialogue content[S2]Speaker 2 dialogue content[S1]...",
    "prompt_audio": "path/to/shared_reference_audio.wav",
    "prompt_text": "[S1]Reference text for speaker 1[S2]Reference text for speaker 2"
}]
# Try to use the ExtractorIterator as an iterator
print("Trying iterator approach...", flush=True)
inputs = processor(data, use_normalize=True)
token_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
text, audios = processor.batch_decode(token_ids)
if not os.path.exists("outputs/"):
    os.mkdir("outputs/")
for i, data in enumerate(audios):
    for j, fragment in enumerate(data):
        print(f"Saving audio_{i}_{j}.wav...", flush=True)
        torchaudio.save(f"outputs/audio_{i}_{j}.wav", fragment.cpu(), 24000)
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