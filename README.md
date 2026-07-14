# SVF-Net

PyTorch implementation of the Structural Vital Force Transformer v0.1 described in `结构生力神经网络设计.md`.

## Run

```powershell
pip install torch
python train.py --steps 10
```

The current script uses random token data only as a smoke test. Replace the batch construction in `train.py` with a real tokenizer/dataset before evaluating language-model quality.
