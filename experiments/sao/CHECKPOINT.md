# Obtaining the Stable Audio Open 1.0 checkpoint

Fine-tunes need the **unwrapped** SAO 1.0 weights.

1. Accept the license on Hugging Face: https://huggingface.co/stabilityai/stable-audio-open-1.0
2. Download the model (example with `huggingface-cli`):

```bash
huggingface-cli download stabilityai/stable-audio-open-1.0 --local-dir $SPDMX_OUTPUT_DIR/dev/experiments/sao/pretrained
```

3. If the HF repo only provides a wrapped Lightning checkpoint, unwrap it with
   `experiments/sao/stable-audio-tools/unwrap_model.py` (see upstream README).

4. Launch:

```bash
uv run python -m experiments.sao.train --arm all \
  --pretrained-ckpt /path/to/unwrapped.ckpt
```

Matched step budget is `max_steps` in [`config.yaml`](config.yaml).
