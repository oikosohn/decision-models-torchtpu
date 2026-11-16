# Decision models on TorchTPU

[English](README.md) · [한국어](README.ko.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

基于 TorchTPU 实现的开源 System One 决策模型，并在训练时间、成本和准确率上进行基准测试。

> System One 决策模型只需一次前向传播，即可从给定选项中选出一个或为其打分，并同时返回该判断经过校准的置信度。

| 模型 | 路径 | 说明 | 笔记本 |
|---|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | [`laya/`](laya/) | - 使用 Google Colab TPU v5e、Google Cloud TPU v5e、v6e<br>- 端到端训练速度至少是两块 T4 GPU 的 1.6 倍 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oikosohn/decision-models-torchtpu/blob/main/laya/notebooks/laya_finetune_typed_decisions_v5e1_colab.ipynb) |

## 许可证

- Laya：[Apache-2.0](laya/LICENSE)