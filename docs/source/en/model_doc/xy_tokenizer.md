# XY Tokenizer

## Overview

The XY Tokenizer model is a specialized tokenizer for multi-modal processing, designed to handle both text and speech tokens in a unified manner.

This model provides efficient tokenization for multi-modal inputs and can be used in conjunction with speech synthesis and processing models.

## Usage

```python
from transformers import XYTokenizerConfig, XYTokenizerModel, XYTokenizerFeatureExtractor
import torch

# Initialize the model
config = XYTokenizerConfig()
model = XYTokenizerModel(config)
feature_extractor = XYTokenizerFeatureExtractor()

# Process multi-modal input
input_data = torch.randn(1, 2, 1000)  # Example multi-modal input
outputs = model(input_data)
```

## XYTokenizerConfig

[[autodoc]] XYTokenizerConfig

## XYTokenizerModel

[[autodoc]] XYTokenizerModel
    - forward

## XYTokenizerFeatureExtractor

[[autodoc]] XYTokenizerFeatureExtractor
    - __call__

## ExtractorIterator

[[autodoc]] ExtractorIterator
