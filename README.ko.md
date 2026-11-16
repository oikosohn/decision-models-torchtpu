# Decision models on TorchTPU

[English](README.md) · [한국어](README.ko.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

TorchTPU 기반 오픈소스 System One 의사결정 모델 및 시간, 비용, 정확도 벤치마크

> System One 의사결정 모델은 순전파 한 번으로 주어진 선택지 중 하나를 고르거나 점수를 매기고, 그 판단에 대한 보정된 신뢰도까지 함께 반환합니다.




| 모델                                            | 경로               | 설명 | 노트북 |
| --------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------- | ---- |
| [Laya](https://github.com/NandhaKishorM/laya) | [`laya/`](laya/) | - Google Colab TPU v5e, Google Cloud TPU v5e, v6e 사용<br>- GPU T4 2장 대비 학습 시간 1.6배 이상 단축 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oikosohn/decision-models-torchtpu/blob/main/laya/notebooks/laya_finetune_typed_decisions_v5e1_colab.ipynb) |


## 라이선스

- Laya: [Apache-2.0](laya/LICENSE).