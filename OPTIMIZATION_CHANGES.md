# Notebook Optimization Changes

## Summary of Changes to mistui-submission-v2-2.ipynb

This document describes the optimizations made to achieve faster execution (target: <15 minutes) and enable multi-GPU training.

### 1. Multi-GPU Training Support

#### LSTM Models (Cells 26, 27)
- Added `torch.nn.DataParallel` wrapper for multi-GPU training
- Automatically detects number of available GPUs
- Scales batch size proportionally with GPU count
- Added `pin_memory=True` and `num_workers=2` to DataLoader for faster data transfer

#### XGBoost (Cell 23)
- Added conditional GPU support via `tree_method='gpu_hist'` and `predictor='gpu_predictor'`
- GPU settings only applied when CUDA is available

### 2. Model Checkpointing

#### New Cell 22: Model Loading
- Checks for pre-trained models before training
- Loads cached models if available (XGBoost, LightGBM, LSTM)
- Sets flags to skip training when models exist

#### Model Saving
- **XGBoost/LightGBM**: Saved using `joblib` to `best_xgb_model.pkl` and `best_lgb_model.pkl`
- **LSTM**: Saved using `torch.save` to `lstm_model.pth` and `optuna_lstm_model.pth`
- Handles DataParallel wrapped models correctly

### 3. Training Speed Optimizations

#### Reduced Training Iterations
- **Optuna XGBoost/LightGBM trials**: 20 → 10 trials
- **Optuna LSTM trials**: 10 → 5 trials
- **LSTM epochs**: 30 → 20 epochs (main training)
- **Optuna LSTM final epochs**: 50 → 30 epochs

#### Increased Batch Sizes
- **LSTM batch size**: 25 → 64 (base)
- Auto-scales to `64 * n_gpus` when multiple GPUs available

#### DataLoader Optimizations
- Added `num_workers=2` for parallel data loading
- Added `pin_memory=True` for faster GPU transfer

### 4. Conditional Training

All training cells now wrapped with conditionals:
- Training skips if cached models exist
- Models are loaded from disk instead of retraining
- Significant time savings on subsequent runs

### Expected Performance Improvements

**First Run (no cached models):**
- Multi-GPU: ~2x faster with 2 GPUs
- Reduced iterations: ~40% faster
- Total improvement: ~60-70% faster

**Subsequent Runs (with cached models):**
- Training completely skipped
- Only preprocessing and prediction runs
- Expected runtime: <5 minutes

### Usage Notes

1. **First Run**: Will train all models and save them
2. **Subsequent Runs**: Will load cached models automatically
3. **Retraining**: Delete `.pkl` and `.pth` files to force retraining
4. **GPU Requirement**: Requires CUDA-compatible GPU(s) for optimal performance

### Files Generated

- `best_xgb_model.pkl` - Trained XGBoost model
- `best_lgb_model.pkl` - Trained LightGBM model  
- `lstm_model.pth` - Trained LSTM model (Cell 26)
- `optuna_lstm_model.pth` - Optuna-tuned LSTM model (Cell 27)
