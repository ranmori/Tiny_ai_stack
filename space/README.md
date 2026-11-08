---
title: Tiny Shakespeare GPT
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
short_description: A 10.8M character-level GPT that writes Shakespeare
python_version: "3.12"
license: mit
---

# Tiny Shakespeare GPT

A 10.8 million parameter character-level GPT, built from scratch and trained on Shakespeare's plays.

- Val loss 1.2751 (the end of *The Taming of the Shrew* and *The Tempest*), test loss 1.3370 (Shakespeare's poems, never trained on or used to choose settings).
- Generation uses a KV cache and runs on the CPU.
- The model weights are downloaded from the [GitHub release](https://github.com/ranmori/Tiny_ai_stack/releases/tag/v1.0) when the Space starts.

Code, training notes and every experiment: [github.com/ranmori/Tiny_ai_stack](https://github.com/ranmori/Tiny_ai_stack)
