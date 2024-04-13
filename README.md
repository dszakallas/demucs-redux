# demucs-redux

> [!WARNING]
> This is a work in progress.

TorchScript compatible inference-only version of [adefossez/demucs](https://github.com/adefossez/demucs).

## Installation

```bash
poetry install
```


### Exporting demucs model to TorchScript

Note that not all models are supported yet.

```bash
poetry run ptexport https://dl.fbaipublicfiles.com/hybrid_transformer/955717e8-8726e21a.th exported/htdemucs.pt
```

