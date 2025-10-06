# Mitsui Commodity Price Modeling - GPU Optimization

This repository contains GPU-optimized machine learning models for commodity price prediction, specifically configured for Kaggle's GPU T4 x2 accelerator.

## 🚀 Quick Start

### For Kaggle Users
1. **Upload** `mistui-submission-v2-2.ipynb` to Kaggle
2. **Enable** GPU T4 x2 accelerator in notebook settings
3. **Add** the Mitsui Commodity Prediction Challenge dataset
4. **Run All** cells
5. **Download** `submission.parquet` from output
6. **Submit** to competition

📖 See [KAGGLE_USAGE_GUIDE.md](KAGGLE_USAGE_GUIDE.md) for detailed instructions.

## ⚡ Performance Improvements

Training time with **GPU T4 x2** vs **CPU**:

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| XGBoost Optuna | 20-50 min | 2-5 min | **5-10x** |
| LightGBM Optuna | 10-30 min | 2-5 min | **3-8x** |
| LSTM Optuna | 60-120 min | 5-10 min | **10-20x** |
| LSTM Training | 20-40 min | 2-5 min | **10-20x** |
| **Total** | **2-4 hours** | **15-30 min** | **10-20x** |

## 📋 What Was Optimized

### Cell 22 - XGBoost & LightGBM
- ✅ Added `tree_method='gpu_hist'` for XGBoost
- ✅ Added `device='cuda'` for LightGBM
- ✅ Increased Optuna trials: 20 → 30

### Cell 26 - LSTM Neural Network
- ✅ Added PyTorch GPU device detection
- ✅ Moved model and data to GPU with `.to(device)`
- ✅ Increased batch size: 32 → 64
- ✅ Increased Optuna trials: 10 → 20
- ✅ Added GPU info logging

### Cell 35 - Submission Generation
- ✅ Made `submission.parquet` generation **MANDATORY**
- ✅ Added error handling for missing data
- ✅ Added file verification after generation
- ✅ Enhanced logging for troubleshooting

## 📁 Files

- **mistui-submission-v2-2.ipynb** - Main notebook (GPU-optimized)
- **GPU_OPTIMIZATION_SUMMARY.md** - Technical details of optimizations
- **KAGGLE_USAGE_GUIDE.md** - Step-by-step deployment guide
- **README.md** - This file

## 🎯 Competition Requirements

The notebook generates `submission.parquet` with:
- `date_id`: Test date identifiers
- `target_0` through `target_N`: Predicted commodity prices

This file is **mandatory** and the notebook will raise errors if it cannot be generated.

## 🛠️ Technical Details

### GPU Optimizations Applied

**XGBoost:**
```python
'tree_method': 'gpu_hist',
'gpu_id': 0
```

**LightGBM:**
```python
'device': 'cuda',
'gpu_platform_id': 0,
'gpu_device_id': 0
```

**PyTorch LSTM:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
batch_X.to(device)
```

### Requirements
All requirements are pre-installed on Kaggle with GPU accelerator:
- XGBoost (with GPU support)
- LightGBM (with GPU support)
- PyTorch (with CUDA)
- Optuna
- Polars
- Pandas
- Scikit-learn

## 📊 Model Architecture

The notebook implements an ensemble model combining:
1. **XGBoost** - Gradient boosting with GPU acceleration
2. **LightGBM** - Gradient boosting with GPU acceleration
3. **LSTM** - Deep learning time series model with GPU acceleration

All models use Optuna for hyperparameter optimization with TimeSeriesSplit cross-validation.

## 🔍 Troubleshooting

### GPU Not Being Used
- Check notebook settings → Accelerator → Select "GPU T4 x2"
- Restart kernel and run again
- Check Cell 26 output for "Using device: cuda"

### CUDA Out of Memory
- Reduce batch size in Cell 26: `batch_size=64` → `batch_size=32`
- Reduce LSTM hidden_size options

### submission.parquet Not Generated
- Check error messages (ValueError or FileNotFoundError)
- Ensure test data is properly added from competition
- Verify all previous cells executed successfully

See [KAGGLE_USAGE_GUIDE.md](KAGGLE_USAGE_GUIDE.md) for more troubleshooting tips.

## 📚 Documentation

- **[GPU_OPTIMIZATION_SUMMARY.md](GPU_OPTIMIZATION_SUMMARY.md)** - Detailed technical documentation of GPU optimizations, expected performance, and implementation details
- **[KAGGLE_USAGE_GUIDE.md](KAGGLE_USAGE_GUIDE.md)** - Complete guide for running the notebook on Kaggle, including setup, expected outputs, and troubleshooting

## 🏆 Competition

This notebook is designed for the **Mitsui Commodity Prediction Challenge** on Kaggle.

Competition Link: [Mitsui Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)

## 📝 License

This project is part of the Mitsui Commodity Prediction Challenge on Kaggle.

## 👥 Contributors

Optimized for GPU acceleration and mandatory submission file generation.

---

**Ready to submit?** Upload to Kaggle, enable GPU, and click Run All! 🚀
