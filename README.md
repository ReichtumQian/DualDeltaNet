# Dual DeltaNet

Implementation of Dual DeltaNet (DDN), built on [FLA](https://github.com/fla-org/flash-linear-attention) and [Flame](https://github.com/fla-org/flame). Includes 130M, 340M, and 1B configurations for DDN, KDA, GDN-2, and Mamba-3 SISO and MIMO.

## Installation

Requires Linux, Python 3.10+, an NVIDIA GPU with compute capability 8.0 or higher, and a CUDA toolkit. Install CUDA-compatible PyTorch 2.7+ first, then run from the repository root:

```bash
python -m pip install -e .
```

Mamba-3 baselines additionally require [`mamba-ssm`](https://github.com/state-spaces/mamba#installation) with Mamba-3 support.

## Training

```bash
bash scripts/train_ddn_130M.sh
```

[scripts](scripts) provides launchers for `ddn`, `kda`, `gdn2`, `mamba3_siso`, and `mamba3_mimo`. Model configurations are in [configs](configs).

The launchers default to single-node training on **8 GPUs**, with settings intended for H200. The sequence length is **2,048 tokens** and the global batch size is **256 sequences**.

By default, training streams the `sample-100BT` subset of `HuggingFaceFW/fineweb-edu` from Hugging Face, shuffles the data, and tokenizes it online with `fla-hub/transformer-1.3B-100B`. This requires network access. Set `TOKENIZER` to use a local tokenizer, or set `DATASET`, `DATASET_NAME`, and `--training.data_files` to use local data files.

Override GPU count and batch size as needed:

```bash
NGPU=1 MICRO_BATCH_SIZE=16 bash scripts/train_ddn_130M.sh
```

Gradient accumulation is calculated automatically. Set `OUTPUT` to choose an output directory.

## License

DDN is released under the [MIT License](LICENSE). FLA and Flame retain their respective [FLA](fla/LICENSE) and [Flame](flame/LICENSE) licenses.
