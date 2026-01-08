# Data Requirements for Testing

## HuggingFace Datasets Needed for Streaming Training

To test the streaming training pipeline, we need access to the following HuggingFace datasets:

### Required for Stage 1 (Mid-Training)

1. **CommitPack** (Code-Flow Learning)
   - Dataset: `bigcode/commitpack-subset-cf`
   - Purpose: Repository evolution patterns, commit transitions
   - Format: `<commit_before>code_before<commit_message>message<commit_after>code_after`
   - Usage: 50-70% of Stage 1 data mix
   - **Status**: ✅ Public dataset, streaming available

2. **Long-Context Repo Snapshots**
   - Dataset: Custom or derived from code repositories
   - Purpose: Long-context repository reasoning
   - Format: Packed repository snapshots
   - Usage: 15-25% of Stage 1 data mix
   - **Status**: ⚠️ May need to create or use alternative

3. **Synthetic Tool Traces**
   - Dataset: Generated from Stage 5 (InfTool loop)
   - Purpose: Tool-use patterns
   - Format: Tool trace format
   - Usage: 10-25% of Stage 1 data mix
   - **Status**: ✅ Generated internally

### Required for Stage 2 (SFT)

1. **High-Quality SWE Trajectories**
   - Dataset: Custom or from SWE-bench style datasets
   - Purpose: Supervised fine-tuning on SWE tasks
   - Format: `(spec → plan → tool calls → patch → tests → summary)`
   - **Status**: ⚠️ Need to identify or create dataset

### Optional Datasets

1. **Code Generation Datasets**
   - `bigcode/the-stack` (subset)
   - `bigcode/starcoderdata`
   - Purpose: Additional code training data

2. **Tool-Use Datasets**
   - Custom MCP tool traces
   - Purpose: Tool calling patterns

## Testing Streaming Pipeline

### When Ready to Test

Run the following to test streaming:

```bash
# 1. Bootstrap environment (Stage 0)
python main.py --stage 0

# 2. Test streaming data loading
python -c "
from streaming.hf_stream import StreamingDataLoader
loader = StreamingDataLoader('bigcode/commitpack-subset-cf', streaming=True)
for i, sample in enumerate(loader.stream(max_samples=10)):
    print(f'Sample {i}: {sample.keys()}')
    if i >= 5:
        break
"

# 3. Test Stage 1 with streaming
python main.py --stage 1 --config configs/stage1_midtrain.yaml --dry-run
```

### Expected Behavior

- ✅ Data streams without downloading full dataset
- ✅ No dataset files materialized in cache
- ✅ Ephemeral cache only (cleaned on exit)
- ✅ Multiple datasets mixed according to ratios

### Verification

Check that:
1. `HF_DATASETS_CACHE` points to ephemeral location (`/tmp/...`)
2. No `.arrow` or `.parquet` files in cache
3. Data streams successfully
4. Memory usage stays reasonable

## Current Status

**Ready for Testing**: ✅
- Streaming infrastructure complete
- Stage 1 ready to test with CommitPack
- Need to verify dataset access

**Action Required**: 
- Test streaming with `bigcode/commitpack-subset-cf`
- Verify no storage occurs
- Check data format compatibility

## Next Steps

1. **Test Streaming** (Ready Now)
   - Test with CommitPack dataset
   - Verify no-storage guarantee
   - Check data format

2. **Stage 1 Training** (After Streaming Verified)
   - Run full Stage 1 with streaming data
   - Monitor safeguards
   - Check anchor regression

3. **Additional Datasets** (As Needed)
   - Identify SWE trajectory datasets
   - Set up long-context repo snapshots
   - Configure synthetic data generation

---

**When you're ready to test streaming, let me know and I'll help verify the data pipeline!**

