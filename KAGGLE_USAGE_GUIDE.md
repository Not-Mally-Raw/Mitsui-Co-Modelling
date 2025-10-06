# Kaggle Usage Guide - mistui-submission-v2-2.ipynb

## Quick Start on Kaggle

### Step 1: Upload Notebook to Kaggle
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook" → "Import Notebook"
3. Upload `mistui-submission-v2-2.ipynb`

### Step 2: Enable GPU Accelerator
1. Click "Settings" in the right sidebar
2. Under "Accelerator", select **"GPU T4 x2"**
3. Click "Save"

### Step 3: Add Competition Data
1. Click "Add Data" in the right sidebar
2. Search for "Mitsui Commodity Prediction Challenge"
3. Add the competition dataset

### Step 4: Run the Notebook
1. Click "Run All" or execute cells sequentially
2. Monitor GPU usage in the right sidebar
3. Wait for training to complete (~15-30 minutes)

### Step 5: Download Submission File
1. After notebook completes, look for `submission.parquet` in the output
2. Click on `submission.parquet` to download
3. Submit to the competition

## Expected Output

### Cell 22 (XGBoost/LightGBM) Output:
```
===== OPTUNA HYPERPARAMETER OPTIMIZATION (GPU-Accelerated) =====
Training shape: X=(rows, features), y=(rows, targets)

1. Optimizing XGBoost hyperparameters with GPU...
2. Optimizing LightGBM hyperparameters with GPU...

✓ XGBoost best params: {...}
  Best CV RMSE: X.XXXXXX

✓ LightGBM best params: {...}
  Best CV RMSE: X.XXXXXX

✓ Optuna hyperparameter tuning completed with GPU acceleration!
```

### Cell 26 (LSTM) Output:
```
Setting up Optuna optimization for LSTM with GPU support...
Using device: cuda
GPU: Tesla T4
GPU Memory: 15.00 GB

Optimizing LSTM hyperparameters with GPU...
[Progress bar showing 20 trials]

✓ LSTM best params: {...}
  Best RMSE: X.XXXXXX

Training final LSTM model on GPU...
Epoch 10/50, Loss: X.XXXXXX
Epoch 20/50, Loss: X.XXXXXX
Epoch 30/50, Loss: X.XXXXXX
Epoch 40/50, Loss: X.XXXXXX
Epoch 50/50, Loss: X.XXXXXX

✓ LSTM training completed with GPU acceleration
```

### Cell 35 (Submission) Output:
```
===== GENERATING SUBMISSION (MANDATORY) =====

1. Loading test data...
   Test data loaded: (rows, columns)

2. Applying feature engineering to test data...
3. Scaling test data with training scaler...
4. Generating ensemble predictions...
5. Creating submission file (MANDATORY)...
   ✓ No null values in submission

6. Saving submission.parquet (MANDATORY)...
✓✓✓ SUBMISSION SAVED: submission.parquet ✓✓✓
   Shape: (rows, columns)
   Columns: ['date_id', 'target_0', ..., 'target_N']

✓ Verified: submission.parquet exists (XX.XX KB)

✓✓✓ SUBMISSION PIPELINE COMPLETED SUCCESSFULLY ✓✓✓
✓✓✓ submission.parquet file is ready for submission ✓✓✓
```

## GPU Verification

To verify GPU is being used:

1. **Check LSTM Cell Output:**
   - Should show "Using device: cuda"
   - Should display GPU name (Tesla T4)
   - Should show GPU memory

2. **Monitor Kaggle GPU Usage:**
   - Look at the right sidebar during training
   - GPU utilization should be 60-90%
   - GPU memory should be actively used

3. **Training Speed:**
   - XGBoost: Should complete in 2-5 minutes
   - LightGBM: Should complete in 2-5 minutes
   - LSTM: Should complete in 5-10 minutes
   - If taking much longer, GPU may not be enabled

## Troubleshooting

### Issue: "Using device: cpu" instead of "cuda"
**Solution:**
- Go to Settings → Accelerator → Select "GPU T4 x2"
- Restart kernel and run again

### Issue: CUDA out of memory error
**Solution:**
- Reduce batch size in Cell 26 (line with `batch_size=64` → try `batch_size=32`)
- Reduce hidden_size options in LSTM hyperparameters

### Issue: submission.parquet not created
**Solution:**
- Check error message - should raise ValueError or FileNotFoundError
- Ensure test data is properly added from competition
- Check that all previous cells executed successfully

### Issue: XGBoost not using GPU
**Solution:**
- XGBoost GPU support is pre-installed on Kaggle
- Check output for any warnings about tree_method
- Try: `import xgboost; print(xgboost.__version__)`

### Issue: LightGBM not using GPU
**Solution:**
- LightGBM GPU support is pre-installed on Kaggle
- Check output for any warnings about device
- Try: `import lightgbm; print(lightgbm.__version__)`

## Performance Benchmarks

### With GPU T4 x2:
- Total training time: ~15-30 minutes
- XGBoost Optuna (30 trials): ~2-5 minutes
- LightGBM Optuna (30 trials): ~2-5 minutes
- LSTM Optuna (20 trials): ~5-10 minutes
- Final LSTM training (50 epochs): ~2-5 minutes

### Without GPU (CPU only):
- Total training time: ~2-4 hours
- XGBoost Optuna: ~20-50 minutes
- LightGBM Optuna: ~10-30 minutes
- LSTM Optuna: ~60-120 minutes
- Final LSTM training: ~20-40 minutes

**Speedup: 10-20x faster with GPU!**

## Submission to Competition

After `submission.parquet` is generated:

### Option 1: Manual Submission
1. Download `submission.parquet` from notebook output
2. Go to competition page
3. Click "Submit Predictions"
4. Upload `submission.parquet`
5. Add submission message: "Ensemble v2 GPU-Optimized"

### Option 2: API Submission (optional)
Add and run a new cell:
```python
!kaggle competitions submit -c mitsui-commodity-prediction-challenge \
    -f submission.parquet \
    -m "Ensemble v2 GPU-Optimized"
```

## Additional Resources

- [Kaggle GPU Documentation](https://www.kaggle.com/docs/notebooks#gpu)
- [XGBoost GPU Documentation](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [LightGBM GPU Documentation](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)

## Notes

- GPU accelerator is free on Kaggle (30 hours/week limit)
- T4 x2 means 2x Tesla T4 GPUs available
- This notebook uses single GPU (gpu_id=0)
- GPU quota resets weekly
- Save your work frequently - sessions timeout after inactivity

---

**Ready to run?** Upload to Kaggle, enable GPU T4 x2, and click "Run All"! 🚀
