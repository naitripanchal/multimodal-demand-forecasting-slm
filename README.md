# A Predictive Supply Chain Framework: Leveraging Small Language Models for Sentiment-Based Demand Forecasting

> **Research Paper Implementation** | MIT World Peace University, Pune

[![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Colab-orange.svg)](https://kaggle.com)
[![Model](https://img.shields.io/badge/Model-Phi--3--mini%204bit-green.svg)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📄 Paper Abstract

Traditional demand forecasting struggles with intermittency, high-frequency noise, and sparse data — problems that purely numerical models cannot fully address. This project proposes a **multimodal late-fusion framework** that combines a Small Language Model (SLM) with historical time-series data to enhance forecasting accuracy.

Using **Microsoft Phi-3-mini (4-bit NF4 quantized)**, we derive a semantic *Market Appeal Index* from unstructured Amazon product metadata and fuse it with lag-1 weekly sales data in a **Random Forest Regressor**.

### Key Results

| Category | Baseline MAE | Multimodal MAE | Change |
|---|---|---|---|
| All Beauty (Hedonic) | 4.59 | 4.46 | **−2.83% ✅** |
| Electronics (Utilitarian) | 3.20 | 3.31 | +3.44% ❌ |

**Insight:** Sentiment is a meaningful demand signal only in emotion-driven (hedonic) product categories. For utilitarian goods, it acts as noise.

---

## 🏗️ Architecture

```
Raw Data (Amazon 2023)
        │
        ▼
Preprocessing (Volume Threshold > 15 units)
        │
        ├──────────────────────────┐
        ▼                          ▼
Qualitative Modality          Quantitative Modality
(Phi-3 SLM Scoring)           (Lag-1 Weekly Sales)
        │                          │
        └──────────┬───────────────┘
                   ▼
          Multimodal Fusion
          (Random Forest Regressor)
                   │
                   ▼
          Final Forecast (Weekly Volume)
```

---

## 📁 Repository Structure

```
.
├── supply_chain_slm_final.ipynb   # Main implementation notebook
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## 🚀 Getting Started

### Run on Kaggle (Recommended)

This notebook is optimized for the **Kaggle T4 GPU** environment (16GB GDDR6 VRAM).

1. Upload `supply_chain_slm_final.ipynb` to [Kaggle Notebooks](https://www.kaggle.com/notebooks)
2. Enable **GPU T4 x2** in notebook settings
3. Add the dataset: [Amazon Product Research 2023](https://www.kaggle.com/datasets) — attach as `multimodal-time-series-forecasting`
4. Run all cells

### Run on Google Colab

1. Open via the badge at the top of the notebook
2. Set Runtime → **T4 GPU**
3. Update dataset paths (see Cell 2 in the notebook)
4. Run all cells

### Local Setup

```bash
pip install -r requirements.txt
jupyter notebook supply_chain_slm_final.ipynb
```

> ⚠️ Requires a GPU with at least **4GB VRAM** for 4-bit quantized inference.

---

## 📦 Dependencies

```
transformers==4.40.2   # Pinned for Phi-3-mini / DynamicCache compatibility
bitsandbytes
accelerate
scikit-learn
pandas
numpy
matplotlib
```

---

## 🔬 Methodology Overview

| Step | Description | Paper Section |
|---|---|---|
| **Data** | Amazon 2023 — All Beauty & Electronics (200 SKUs/category) | §II-B |
| **Preprocessing** | Volume threshold filter (Σ demand > 15 units) | §II-C |
| **Temporal Alignment** | Daily → Weekly aggregation (low-pass filter) | §II-D |
| **SLM Inference** | Phi-3-mini zero-shot → Market Appeal Index (1–10) | §II-E |
| **Feature Fusion** | Late fusion: Lag-1 sales + SLM score | §II-F |
| **Model** | Random Forest Regressor (100 estimators) | §II-G |
| **Evaluation** | Ablation study — Baseline vs. Multimodal (MAE) | §II-H |

---

## 🧠 Key Technical Innovations

- **4-bit NF4 Quantization**: Reduces Phi-3-mini from ~15GB (FP16) to ~2.2GB — runs on commodity hardware
- **Zero-Shot Prompting**: No fine-tuning required; the SLM acts as a retail analyst via instruction prompting
- **Temporal Aggregation as Low-Pass Filter**: Daily→Weekly smoothing is the critical enabler of multimodal lift
- **Edge-Deployability**: Entire pipeline runs on a single Kaggle T4 GPU — no data-center scale required

---

## 👥 Authors

- **Bhavana Tiple** — bhavana.tiple@mitwpu.edu.in
- **Riya Kulkarni** — riyakulkarni189@gmail.com
- **Naitri Panchal** — naitripanchal0987@gmail.com
- **Heramb Churi** — heramb.v.churi@gmail.com

*Department of Computer Science and Engineering, MIT World Peace University, Pune, India*

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{tiple2025predictive,
  title={A Predictive Supply Chain Framework: Leveraging Small Language Models for Sentiment-Based Demand Forecasting},
  author={Tiple, Bhavana and Kulkarni, Riya and Panchal, Naitri and Churi, Heramb},
  institution={MIT World Peace University},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License.
