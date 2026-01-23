# BP+OSD Decoder Configuration for p=0.001 Target

## Changes Made

### 1. Dataset Update
- **Changed from**: `sc_d3_r3_p0010_z` (p=0.010)
- **Changed to**: `sc_d3_r3_p0001_z` (p=0.001)
- Dataset contains 10,000 syndrome samples

### 2. BP Iterations Increased
- **Changed from**: `BP_MAX_ITER = 20`
- **Changed to**: `BP_MAX_ITER = 50`
- Rationale: More iterations needed for convergence at lower error rates

### 3. OSD Order
- **Maintained**: `OSD_ORDER = 15` (searches 2^15 = 32,768 candidates)
- This provides good balance between accuracy and runtime

### 4. Test Configuration
- **Samples**: 1000 (for reasonable runtime ~1.6 hours)
- **Batch size**: 50
- **Device**: CUDA (GPU acceleration enabled)

## Target Performance

According to paper http://arxiv.org/abs/2005.07016:
- **Target logical error rate**: 0.0001 (0.01%)
- **Code**: d=3 rotated surface code
- **Physical error rate**: p=0.001

## Current Status

The decoder is running with the updated configuration:
- File: `run_batch_demo.py`
- Output log: `decoder_output_1000.log`
- Monitor script: `monitor_decoder.sh`

### Preliminary Results (100 samples)
- BP-only: 0.0000 (0 errors)
- BP+OSD-15: 0.0100 (1 error)

### Expected Runtime
- ~6 seconds per sample
- 1000 samples: ~1.6 hours
- Full 10,000 samples: ~16 hours

## Monitoring Progress

To monitor the decoder progress:
```bash
./monitor_decoder.sh
```

Or check the log directly:
```bash
tail -f decoder_output_1000.log
```

## Next Steps

1. Wait for 1000-sample run to complete
2. If logical error rate is close to target (0.0001), consider running with more samples
3. If error rate is higher, may need to:
   - Increase OSD order (e.g., OSD-20 or OSD-25)
   - Further increase BP iterations
   - Adjust damping parameter

## Files Modified

- `run_batch_demo.py`: Updated dataset path, BP_MAX_ITER, and NUM_SAMPLES
