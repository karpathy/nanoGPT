# Coverage Improvement Opportunities

**Current Global Coverage**: 87.28%  
**Coverage Threshold**: 87.00% (raised from 79.00%)  
**Last Updated**: 2025-10-05

## Summary

After the comprehensive coverage raising session (PR #49), we've achieved 87.28% global coverage, up from 80.73% (+6.55%). This document outlines remaining opportunities to reach 90%+ coverage.

## Modules by Coverage Tier

### ✅ Complete (100% Coverage) - 11 Modules

These modules have achieved 100% coverage:

1. `ml_playground/data_pipeline/preparer.py`
2. `ml_playground/configuration/loading.py`
3. `ml_playground/training/hooks/evaluation.py`
4. `ml_playground/training/ema.py`
5. `ml_playground/training/hooks/components.py`
6. `ml_playground/experiments/bundestag_tiktoken/preparer.py`
7. `ml_playground/training/hooks/logging.py`
8. `ml_playground/experiments/speakger/preparer.py`
9. `ml_playground/experiments/bundestag_char/preparer.py`

### 🎯 High-Value Opportunities (86-95% Coverage)

These modules are close to complete and offer the best ROI for coverage improvement:

#### 1. `ml_playground/training/checkpointing/service.py` - 94.12%
- **Missing**: 2 lines (defensive code paths)
- **Opportunity**: Exception handling in DI override paths (pragma: no cover)
- **Effort**: Low
- **Impact**: Minimal (defensive code)

#### 2. `ml_playground/training/checkpointing/checkpoint_manager.py` - 93.17%
- **Missing**: 16 lines (initialization edge cases)
- **Lines**: 78-81, 89-90, 97-100, 103-106, 266, 365, 424-442
- **Opportunity**: 
  - FileNotFoundError handling in `_path_unlink` (78-81)
  - PosixPath import edge cases (89-90)
  - `missing_ok` parameter probe failures (97-100, 103-106)
  - Best checkpoint filename validation (266)
  - Sidecar file cleanup (365)
  - Safe globals handling in load_best_checkpoint (424-442)
- **Effort**: Medium (requires mocking internal behavior)
- **Impact**: Low (mostly defensive/compatibility code)

#### 3. `ml_playground/data_pipeline/transforms/tokenization.py` - 92.96%
- **Missing**: 3 lines (exception handlers)
- **Lines**: 50-56, 84-93, 88-93, 90-91
- **Opportunity**:
  - WordTokenizer vocab building edge case (50-56)
  - Exception handling in metadata creation (84-93, 90-91)
- **Effort**: Low
- **Impact**: Low (exception handlers)

#### 4. `ml_playground/models/core/inference.py` - 92.98%
- **Missing**: 2 lines
- **Lines**: 15, 53
- **Opportunity**:
  - AttributeError when model has no config (15)
  - AttributeError in generate_tokens (53)
- **Effort**: Very Low (already have tests, just need to hit these specific branches)
- **Impact**: Low (error handling)

#### 5. `ml_playground/experiments/bundestag_qwen15b_lora_mps/preparer.py` - 87.93%
- **Missing**: 5 lines (OSError edge cases)
- **Lines**: 58-59, 78-80
- **Opportunity**: Exception handling in helper functions
- **Effort**: Low
- **Impact**: Low (defensive code)

#### 6. `ml_playground/cli.py` - 86.97%
- **Missing**: 26 lines
- **Lines**: 226, 261-264, 274-291, 301-317, 388-390
- **Opportunity**:
  - CLI argument parsing edge cases
  - Experiment-specific CLI paths
  - Error handling in main entry points
- **Effort**: Medium-High (requires CLI integration testing)
- **Impact**: Medium (user-facing code paths)

### ⚠️ Hardware/Platform-Dependent (Cannot Improve Without Hardware)

#### 7. `ml_playground/training/hooks/runtime.py` - 78.57%
- **Missing**: 5 lines (CUDA initialization)
- **Lines**: 38-42
- **Opportunity**: CUDA device setup and TF32 configuration
- **Blocker**: Requires CUDA-capable GPU hardware
- **Effort**: N/A (hardware-dependent)
- **Impact**: N/A (tested on CUDA systems)

### 📋 Protocol Classes (Tested via Implementations)

These are interface definitions tested through their concrete implementations:

#### 8. `ml_playground/core/tokenizer_protocol.py` - 75.00%
- **Type**: Protocol class (interface definition)
- **Coverage**: Tested via CharTokenizer, WordTokenizer, TiktokenTokenizer
- **Opportunity**: None (protocol methods are abstract)

#### 9. `ml_playground/core/logging_protocol.py`
- **Type**: Protocol class (interface definition)
- **Coverage**: Tested via actual logger implementations
- **Opportunity**: None (protocol methods are abstract)

## Recommended Next Steps

### To Reach 90% Coverage

**Priority 1: Quick Wins (Low Effort, Immediate Impact)**
1. Add 2-3 tests for `cli.py` to cover main CLI paths → +0.5%
2. Add exception handling tests for `tokenization.py` → +0.1%
3. Add error path tests for `inference.py` → +0.1%

**Estimated Result**: ~88% coverage

**Priority 2: Medium Effort**
1. Add CLI integration tests for experiment-specific paths → +0.8%
2. Add edge case tests for `bundestag_qwen15b_lora_mps/preparer.py` → +0.2%

**Estimated Result**: ~89% coverage

**Priority 3: Diminishing Returns**
1. Add complex mocking for `checkpoint_manager.py` initialization → +0.5%
2. Add defensive code path tests for `service.py` → +0.1%

**Estimated Result**: ~89.6% coverage

### To Reach 95% Coverage

Would require:
- Comprehensive CLI integration test suite
- Complex mocking of internal PyTorch/system behavior
- CUDA hardware for runtime.py testing

**Effort**: High  
**ROI**: Low (mostly defensive/edge case code)

## Coverage Philosophy

The project follows a pragmatic approach to coverage:

1. **100% coverage for core business logic** ✅ Achieved
2. **High coverage (>90%) for user-facing APIs** ⚠️ In progress (cli.py)
3. **Reasonable coverage (>85%) for infrastructure** ✅ Achieved
4. **Defensive code may remain untested** ✅ Acceptable (pragma: no cover)
5. **Hardware-dependent code tested on target platforms** ✅ Acceptable

## Conclusion

The current 87.28% coverage represents excellent test coverage with all critical and medium-priority gaps addressed. Reaching 90% is achievable with focused CLI testing. Beyond 90%, returns diminish significantly as remaining gaps are primarily:

- Defensive error handling
- Platform-specific code
- Initialization edge cases
- Protocol definitions

**Recommendation**: Focus on CLI integration testing to reach 88-89%, then reassess based on actual bug patterns and maintenance needs.
