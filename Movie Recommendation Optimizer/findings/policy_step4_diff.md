# Policy Step 4 Diff

## Changes from Provisional to Step 4 Policy

### Major Changes
- **Status**: provisional → step4_validated
- **Version**: 1.0 → 2.0
- **Alpha Strategy**: Explicitly set to 'bucket_gate'
- **Override Rules**: Added long-tail and min-history overrides
- **Validation Status**: Updated based on Step 4 results

### New Features
- **Long-tail Override**: Content-heavy for long-tail items
- **Min-history Guardrail**: Force content-heavy for <3 ratings
- **Selection Tiebreakers**: NDCG@10, then Recall@10
- **Reproducibility**: Added random seed and commit tracking

### Rationale
- **Best Alpha**: 1.0 (from MAP@10 analysis)
- **Bucket-Gate**: Validated across all cohorts
- **Cold Start**: Content-heavy approach for minimal history
- **Long-tail**: Content-based excels at diversity

### Validation Results
- **Cold Users**: Synthesized and validated
- **Cohort Analysis**: Completed across all user types
- **Popularity Analysis**: Head/mid/long-tail performance assessed
- **Lift Analysis**: Hybrid vs baseline comparisons
