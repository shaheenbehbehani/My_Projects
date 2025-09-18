# Step 3c.3 – Patched Production-Lite Evaluation Report

**Generated**: 2025-09-04 12:37:30

## Results Summary

- **Best α**: 0.00
- **Best Recall@10**: 0.0117
- **Content-only Recall@10**: 0.0117
- **Baseline Coverage**: 11.8%

## α=0.0 Cheap Baseline Method

- **Seed limit per user**: S=10 (prefer last-N positives)
- **Neighbors per seed**: k_content_seed=20 from cached neighbors
- **Caps**: C_max_baseline=400 candidates
- **Coverage achieved**: {baseline_coverage:.1%}
- **⚠️ Caveat**: Baseline coverage < 60%

## Detailed Results

| α | Recall@10 | MAP@10 | Users | Coverage | Time (s) | Partial |
|---|-----------|--------|-------|----------|----------|----------|
| 0.00 | 0.0117 | 0.0044 | 118 | 11.8% | 300.9 | No |

## Notes

- α=0.0 uses cheap content baseline to avoid timeouts
- Coverage target: ≥60% with cold+light users prioritized
- Results should be validated on larger samples for production
