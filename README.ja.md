# Decision models on TorchTPU

[English](README.md) · [한국어](README.ko.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

TorchTPU で実装したオープンソースの System One 意思決定モデルです。学習時間、コスト、精度のベンチマークを含みます。

> System One 意思決定モデルは、1 回の順伝播で与えられた選択肢から 1 つを選ぶかスコアを付け、その判断の較正済み信頼度もあわせて返します。

| モデル | パス | 説明 | ノートブック |
|---|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) | [`laya/`](laya/) | - Google Colab TPU v5e、Google Cloud TPU v5e、v6e を使用<br>- T4 GPU 2 枚と比べて学習全体で 1.6 倍以上高速 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oikosohn/decision-models-torchtpu/blob/main/laya/notebooks/laya_finetune_typed_decisions_v5e1_colab.ipynb) |

## ライセンス

- Laya: [Apache-2.0](laya/LICENSE).