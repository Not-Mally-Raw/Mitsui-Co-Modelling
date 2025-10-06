# GPU Optimization Summary for Kaggle T4 x2

This document summarizes the GPU optimizations applied to `mistui-submission-v2-2.ipynb` for faster training on Kaggle's T4 x2 GPU environment.

## Changes Made

### 1. Cell 22 - XGBoost and LightGBM GPU Acceleration

**XGBoost Optimizations:**
- Added `tree_method='gpu_hist'` for GPU-accelerated tree building
- Added `gpu_id=0` to specify GPU device
- Increased `n_trials` from 20 to 30 (faster with GPU)

**LightGBM Optimizations:**
- Added `device='cuda'` for GPU acceleration
- Added `gpu_platform_id=0` and `gpu_device_id=0` for GPU selection
- Increased `n_trials` from 20 to 30 (faster with GPU)

**Expected Performance Improvement:**
- XGBoost: 5-10x faster training on GPU
- LightGBM: 3-8x faster training on GPU

### 2. Cell 26 - LSTM GPU Acceleration

**PyTorch LSTM Optimizations:**
- Added proper device detection: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Added GPU memory and device information logging
- Moved model to GPU: `.to(device)`
- Moved all tensor operations to GPU: `batch_X.to(device)`, `batch_y.to(device)`
- Increased batch size from 32 to 64 (better GPU utilization)
- Increased `n_trials` from 10 to 20 (faster with GPU)
- Added proper CPU/GPU data transfer for predictions

**Expected Performance Improvement:**
- LSTM: 10-20x faster training on GPU

### 3. Cell 35 - Mandatory submission.parquet Generation

**Submission File Improvements:**
- Made submission.parquet generation **MANDATORY** with error handling
- Added fallback to load test data from local dataset if Kaggle input fails
- Added explicit error raising if test data cannot be loaded
- Added validation to ensure submission.parquet file is created
- Added file size verification after creation
- Enhanced logging with clear success messages

**Key Changes:**
- Raises `ValueError` if test data cannot be loaded
- Raises `FileNotFoundError` if submission.parquet is not created
- Marked all submission steps as "(MANDATORY)" in print statements
- Added verification that file exists after writing

## How to Use on Kaggle

1. **Enable GPU Accelerator:**
   - In Kaggle notebook settings, enable "GPU T4 x2" accelerator
   - The notebook will automatically detect and use GPU

2. **Install Dependencies:**
   - XGBoost with GPU support is pre-installed on Kaggle
   - LightGBM with GPU support is pre-installed on Kaggle
   - PyTorch with CUDA is pre-installed on Kaggle

3. **Run the Notebook:**
   - Execute all cells in order
   - GPU acceleration will be used automatically
   - Training will be significantly faster than CPU

4. **Submission File:**
   - `submission.parquet` will be generated in the notebook output
   - File is mandatory and will raise errors if not created
   - Ready for direct submission to Kaggle competition

## Verification

To verify GPU is being used:
- Check Cell 26 output for "Using device: cuda" message
- Check Cell 26 output for GPU name and memory info
- Monitor GPU utilization in Kaggle's resource usage panel

## Expected Training Time

With GPU T4 x2 on Kaggle:
- XGBoost Optuna tuning: ~2-5 minutes (vs 20-50 minutes on CPU)
- LightGBM Optuna tuning: ~2-5 minutes (vs 10-30 minutes on CPU)
- LSTM Optuna tuning: ~5-10 minutes (vs 60-120 minutes on CPU)
- Total training time: ~15-30 minutes (vs 2-4 hours on CPU)

## Troubleshooting

If GPU is not being used:
1. Check Kaggle notebook settings - ensure GPU accelerator is enabled
2. Check for CUDA availability: `torch.cuda.is_available()`
3. Check XGBoost GPU support: `xgboost.__version__`
4. Check LightGBM GPU support: `lightgbm.__version__`

## Competition Requirements

This notebook generates `submission.parquet` which is the required format for the Mitsui Commodity Prediction Challenge on Kaggle. The file includes:
- `date_id`: Test date identifiers
- `target_0` through `target_N`: Predicted commodity prices

The submission file is guaranteed to be created or the notebook will fail with a clear error message.
