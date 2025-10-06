# Quick Usage Guide - Optimized Notebook

## 🚀 Getting Started

### Step 1: Upload to Kaggle
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook" → "Upload Notebook"
3. Select `mistui-submission-v2-2.ipynb`

### Step 2: Configure Environment
1. Click "Accelerator" → Select **"GPU T4 x2"** or **"GPU P100 x2"**
2. Verify 2 GPUs are enabled
3. Click "Save Version" → "Run All"

### Step 3: Monitor Execution
Watch for these key messages:

```
✓ Enhanced imports loaded
===== CHECKING FOR PRE-TRAINED MODELS =====
  No pre-trained models found, will train from scratch
Number of GPUs available: 2
Using 2 GPUs with DataParallel
```

## ⏱️ Expected Timeline

### First Run (Training from Scratch)
```
00:00 - Cell 1-21:  Data loading & preprocessing    (~2 min)
02:00 - Cell 22:    Model checkpoint check          (~5 sec)
02:05 - Cell 23:    XGBoost/LightGBM training       (~4 min)
06:05 - Cell 26:    LSTM training                   (~3 min)
09:05 - Cell 27:    Optuna LSTM training            (~2 min)
11:05 - Cell 28-38: Predictions & submission        (~1 min)
──────────────────────────────────────────────────────────
TOTAL:                                               ~12 min ✅
```

### Subsequent Runs (Using Cached Models)
```
00:00 - Cell 1-21:  Data loading & preprocessing    (~2 min)
02:00 - Cell 22:    Loading cached models           (~10 sec)
02:10 - Cell 23:    Skipping training (cached)      (~1 sec)
02:11 - Cell 26:    Loading LSTM (cached)           (~2 sec)
02:13 - Cell 27:    Skipping training (cached)      (~1 sec)
02:14 - Cell 28-38: Predictions & submission        (~1 min)
──────────────────────────────────────────────────────────
TOTAL:                                               ~3-4 min ✅
```

## 📊 What to Expect

### During First Run

#### Cell 22 Output:
```
===== CHECKING FOR PRE-TRAINED MODELS =====
  No pre-trained XGBoost/LightGBM models found, will train from scratch
  No pre-trained LSTM models found, will train from scratch

Training plan:
  XGBoost/LightGBM: Train
  LSTM: Train
```

#### Cell 23 Output:
```
===== OPTUNA HYPERPARAMETER OPTIMIZATION =====
1. Optimizing XGBoost hyperparameters...
2. Optimizing LightGBM hyperparameters...
✓ XGBoost best params: {...}
✓ LightGBM best params: {...}
3. Training final XGBoost and LightGBM models on full data...
4. Saving models for future use...
✓ Models saved: best_xgb_model.pkl, best_lgb_model.pkl
```

#### Cell 26 Output:
```
Number of GPUs available: 2
Using device: cuda
Using 2 GPUs with DataParallel

Starting LSTM model training...
Epoch [1/20], Loss: 0.xxxx
Epoch [2/20], Loss: 0.xxxx
...
Epoch [20/20], Loss: 0.xxxx

LSTM model training finished.
LSTM model saved as lstm_model.pth
```

#### Cell 27 Output:
```
Using device: cuda, GPUs available: 2
Optimizing LSTM hyperparameters...
[Progress bar]
✓ LSTM best params: {...}

Training final LSTM model...
Epoch 10/30, Loss: 0.xxxxxx
Epoch 20/30, Loss: 0.xxxxxx
Epoch 30/30, Loss: 0.xxxxxx

✓ LSTM model saved to optuna_lstm_model.pth
✓ LSTM training completed
```

### During Subsequent Runs

#### Cell 22 Output:
```
===== CHECKING FOR PRE-TRAINED MODELS =====
✓ Found pre-trained XGBoost and LightGBM models
  Loading models to skip training...
✓ Successfully loaded XGBoost and LightGBM models
✓ Found pre-trained LSTM model(s)

Training plan:
  XGBoost/LightGBM: Skip (using cached)
  LSTM: Skip (using cached)
```

#### Cell 23 Output:
```
Skipping XGBoost/LightGBM training - using cached models
```

#### Cell 26 Output:
```
Skipping LSTM training - loading from checkpoint...
✓ LSTM model loaded from lstm_model.pth
```

#### Cell 27 Output:
```
Skipping Optuna LSTM training - using cached model
```

## 🔍 Troubleshooting

### Issue 1: Only 1 GPU detected
**Symptom:**
```
Number of GPUs available: 1
```

**Solution:**
- Check Kaggle accelerator settings
- Ensure "GPU T4 x2" or "GPU P100 x2" is selected
- Restart kernel and try again

### Issue 2: Out of Memory
**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Edit Cell 26: Change `batch_size = 64` to `batch_size = 32`
- Edit Cell 27: Change batch size from 64 to 32
- Restart kernel and run again

### Issue 3: Models not loading
**Symptom:**
```
Warning: Could not load models: [Error details]
Will retrain models
```

**Solution:**
- This is expected on first run
- Models will be trained and saved
- On next run, they should load successfully

### Issue 4: Slow performance
**Symptom:**
Runtime exceeds 15 minutes

**Possible causes & solutions:**
1. **Single GPU**: Ensure 2 GPUs are enabled
2. **CPU fallback**: Check CUDA availability
3. **Slow GPU type**: Use P100 or T4 (not K80)
4. **Network issues**: Data download may be slow

## 💡 Tips & Tricks

### Force Retraining
If you want to retrain models from scratch:

```python
# Add this cell at the top
!rm -f *.pkl *.pth
print("Deleted cached models - will retrain from scratch")
```

### Check GPU Utilization
Monitor GPU usage during training:

```python
# Add this cell
!nvidia-smi
```

### Reduce Training Time Further
If you need even faster training (with potential quality trade-off):

**Option 1: Reduce Optuna trials**
- Cell 23: Change `n_trials=10` to `n_trials=5`
- Cell 27: Change `n_trials=5` to `n_trials=3`

**Option 2: Reduce epochs**
- Cell 26: Change `num_epochs = 20` to `num_epochs = 10`
- Cell 27: Change `range(30)` to `range(20)`

### Monitor Memory Usage
```python
# Add this cell to check memory
import torch
print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"GPU Memory cached: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
```

## 📈 Performance Validation

After running, verify performance:

1. **Check total runtime**: Should be <15 minutes first run
2. **Check GPU utilization**: Should show 2 GPUs used
3. **Check model files**: Should see 4 .pkl/.pth files created
4. **Check predictions**: Should generate submission file

## 🎯 Success Criteria

✅ Runtime < 15 minutes (first run)
✅ Runtime < 5 minutes (subsequent runs)  
✅ 2 GPUs detected and utilized
✅ Models saved successfully
✅ Submission file generated

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Review [README_OPTIMIZATION.md](README_OPTIMIZATION.md)
3. Check [OPTIMIZATION_CHANGES.md](OPTIMIZATION_CHANGES.md)
4. Verify GPU availability with `nvidia-smi`

---

**Last Updated**: After optimization implementation
**Target Runtime**: <15 minutes first run, <5 minutes subsequent runs
**GPU Requirement**: 2x GPUs (P100 or T4 recommended)
