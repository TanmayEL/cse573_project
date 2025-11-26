## Machine Specs (Run on local machine)

- **CPU**: AMD Ryzen 9 8945HS
- **GPU**: NVIDIA Geforce RTX 4060 Laptop GPU
- **RAM**: 16 GB
- **OS**: Windows 11
- **Python**: 3.12.5
- **PyTorch**:
  - `torch` = 2.2.0+cu118
  - `torchvision` = 0.17.2+cpu
  - `CUDA`: 12.1

## Training
All training happens inside the notebook via the `train(...)` function.

This is the **exact** configuration used:

```python
train(
    epochs=25,
    batch_size=128,
    lr=2e-4,
    steps=400,
    sample_every=400,
    sample_steps=50,
    center_box=12,
    dc_repeats=2,
    dc_fixed_z=True,
    pred="v",
    p2_k=1.0,
    p2_gamma=1.0,
    hole_weight=5.0,
    seed=42,
    clip_grad=1.0,
    ema_decay=0.999,
    warmup_steps=1000,
    self_cond=True,
    eta=0.0,
    coord_conv=True,
    grad_accum=1,
)
```

This will:
- Train the DDIM inpainting model on masked MNIST.
- Periodically save inpainting panels under `outputs/inpaint/panel_*.png`.
- At the end, save a checkpoint to `outputs/inpaint/last.pt`.


## Sampling

After training completes and `outputs/inpaint/last.pt` exists:

Run the cell that calls (exact command used):

```python
sample_cmd(
    ckpt="outputs/inpaint/last.pt",
    n=16,
    steps=50,
    center_box=12,
    dc_repeats=2,
    dc_fixed_z=True,
    pred="v",
    self_cond=True,
    init_from_y=True,
    eta=0.0,
    coord_conv=True,
)
```

This will:
- Load the EMA weights from `outputs/inpaint/last.pt`
- Inpaint a small batch of test MNIST digits
- Save a panel image to `outputs/inpaint/samples.png`

The notebook cell then displays the generated inpainting panel inline.


## Quantitative Evaluation (PSNR / L1 on Hole)

To compute the metrics required for grading:

Run the cell that calls (exact command used):

```python
eval_cmd(
    ckpt="outputs/inpaint/last.pt",
    batch_size=128,
    n_eval=500,
    steps=50,
    center_box=12,
    dc_repeats=2,
    dc_fixed_z=True,
    pred="v",
    self_cond=True,
    init_from_y=True,
    eta=0.0,
    coord_conv=True,
    seed=42,
)
```

This will:
- Run inpainting on up to `n_eval` MNIST test images
- Compute **PSNR** and **L1** **only on the masked (hole) region**
- Save metrics to `results/inpaint_metrics.json`


## Random Seeds

- **Training seed**: `seed=42` (argument to `train(...)`)
- **Evaluation seed**: `seed=42` (argument to `eval_cmd(...)`)


