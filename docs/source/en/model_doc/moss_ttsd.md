# MOSS-TTSD

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

## Usage

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
    "base_path": "./examples",
    "text": "[S1]，你到底能不能好好工作？我劝你一句，这个时代，不跟上AI浪潮，就会被彻底淘汰！[S2]这个嘛，那我得先问问硅基之主",
    "system_prompt": "你是一个根据文本生成对应音频的语音合成器。",
    "prompt_text": "[S1]嘎子，你听叔的，你听叔的，其实你跟所有人PK，有的时候我也在看，我也在看，无非两，两件事，一个是面子，不想输。[S2]你别说，那天潘老师有一个徒弟开直播，给我开专场，潘老师一徒弟开直播给我开专场，给我一顿骂。",
    "prompt_audio": "panchangjiang_gazi.wav",
}]

# Try to use the ExtractorIterator as an iterator
print("Trying iterator approach...", flush=True)
inputs = processor(data)
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
