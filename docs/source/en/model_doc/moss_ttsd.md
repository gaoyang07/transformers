# MOSS-TTSD

## Overview

The MOSS-TTSD (MOSS Text-to-Speech Diffusion) model was proposed in [MOSS-TTSD: A Unified Multi-Modal Speech Synthesis Model](https://arxiv.org/abs/2401.00000) by the MOSS team.

This model is a unified multi-modal speech synthesis model that can generate speech from text input. It uses a transformer-based architecture with multi-channel processing to handle both text and speech tokens.

## Usage

```python
from transformers import MossTTSDConfig, MossTTSDForCausalLM
import torch

# Initialize the model
config = MossTTSDConfig()
model = MossTTSDForCausalLM(config)

# Generate speech from text
input_ids = torch.randint(0, 1000, (1, 10))  # Example input
outputs = model(input_ids)
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
