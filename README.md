# Decision models on TorchTPU

[English](README.md) · [한국어](README.ko.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

Open-source System One decision models implemented with TorchTPU, benchmarked on training time, cost, and accuracy.

> A System One decision model picks one of the given options or scores them in a single forward pass, and returns a calibrated confidence for that judgment along with it.

| Model | Path | Description | Notebook |
|---|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | [`laya/`](laya/) | - Uses Google Colab TPU v5e, Google Cloud TPU v5e, v6e<br>- Trains at least 1.6× faster end to end than two T4 GPUs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oikosohn/decision-models-torchtpu/blob/main/laya/notebooks/laya_finetune_typed_decisions_v5e1_colab.ipynb) |

## License

- Laya: [Apache-2.0](laya/LICENSE).