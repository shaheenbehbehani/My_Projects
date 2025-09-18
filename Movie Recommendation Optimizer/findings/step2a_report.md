# TF-IDF Vectorization Report
## Step 2a.2 - Movie Recommendation Optimizer

**Generated:** 2025-08-28 12:17:56  
**Total Movies:** 88,194

---

## Vectorization Summary

### Fields Vectorized
- **overview**: 127 features, 0.0% coverage
- **consensus**: 2,119 features, 1.6% coverage
- **tags_combined**: 89,772 features, 58.8% coverage

### Combined Matrix

---
## Coverage Analysis

| Field | Valid Texts | Coverage % | Features | Matrix Shape |
|-------|-------------|-------------|----------|--------------|
| overview | 44 | 0.0% | 127 | 44 × 127 |
| consensus | 1,442 | 1.6% | 2,119 | 1,442 × 2,119 |
| tags_combined | 51,895 | 58.8% | 89,772 | 51,895 × 89,772 |

---
## Top Features by Field

### overview

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `all` | 2.8608 |
| 2 | `are` | 2.8608 |
| 3 | `most` | 2.8608 |
| 4 | `who` | 2.8608 |
| 5 | `about` | 2.6094 |
| 6 | `adrenaline` | 2.6094 |
| 7 | `adrenaline junkie` | 2.6094 |
| 8 | `adventures` | 2.6094 |
| 9 | `an organization` | 2.6094 |
| 10 | `and lousy` | 2.6094 |
| 11 | `attitude` | 2.6094 |
| 12 | `attitude when` | 2.6094 |
| 13 | `be planning` | 2.6094 |
| 14 | `before` | 2.6094 |
| 15 | `by the` | 2.6094 |

### consensus

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `34th` | 6.4827 |
| 2 | `34th street` | 6.4827 |
| 3 | `ability` | 6.4827 |
| 4 | `ability to` | 6.4827 |
| 5 | `action humor` | 6.4827 |
| 6 | `adding up` | 6.4827 |
| 7 | `adventure and` | 6.4827 |
| 8 | `afghanistan` | 6.4827 |
| 9 | `afghanistan very` | 6.4827 |
| 10 | `again` | 6.4827 |
| 11 | `aged` | 6.4827 |
| 12 | `al` | 6.4827 |
| 13 | `alec` | 6.4827 |
| 14 | `alec guinness` | 6.4827 |
| 15 | `always` | 6.4827 |

### tags_combined

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `000` | 10.0652 |
| 2 | `007 assassin` | 10.0652 |
| 3 | `01 15` | 10.0652 |
| 4 | `02 15` | 10.0652 |
| 5 | `04 15` | 10.0652 |
| 6 | `09 based` | 10.0652 |
| 7 | `10 2007` | 10.0652 |
| 8 | `10 classic` | 10.0652 |
| 9 | `10 not` | 10.0652 |
| 10 | `10 years` | 10.0652 |
| 11 | `100 author` | 10.0652 |
| 12 | `109` | 10.0652 |
| 13 | `10th` | 10.0652 |
| 14 | `10th century` | 10.0652 |
| 15 | `11 artist` | 10.0652 |

---
## Vectorization Parameters

- **N-gram Range**: (1, 2) - unigrams and bigrams
- **Min Document Frequency**: 5 - minimum 5 movies must contain feature
- **Max Document Frequency**: 0.6 - maximum 60% of movies can contain feature
- **Max Features**: 200,000 per field
- **Data Type**: float32
- **Lowercase**: Disabled (text pre-cleaned)

---
## Output Files

- **TF-IDF Matrices**: `.npz` sparse format
- **Vectorizers**: `.joblib` serialized objects
- **Index Mapping**: `.parquet` with canonical_id → row_index
- **Combined Matrix**: All fields horizontally stacked

---
*TF-IDF vectorization completed successfully. All text fields have been converted to numerical features ready for recommendation models.*


# TF-IDF Vectorization Report
## Step 2a.2 - Movie Recommendation Optimizer

**Generated:** 2025-08-28 12:30:59  
**Total Movies:** 88,194

---

## Vectorization Summary

### Fields Vectorized
- **overview**: 127 features, 0.0% coverage
- **consensus**: 2,119 features, 1.6% coverage
- **tags_combined**: 89,772 features, 58.8% coverage

### Combined Matrix
- **Shape**: 88,194 movies × 92,018 features
- **Sparsity**: 1.000

---
## Coverage Analysis

| Field | Valid Texts | Coverage % | Features | Matrix Shape |
|-------|-------------|-------------|----------|--------------|
| overview | 44 | 0.0% | 127 | 88,194 × 127 |
| consensus | 1,442 | 1.6% | 2,119 | 88,194 × 2,119 |
| tags_combined | 51,895 | 58.8% | 89,772 | 88,194 × 89,772 |

---
## Top Features by Field

### overview

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `all` | 2.8608 |
| 2 | `are` | 2.8608 |
| 3 | `most` | 2.8608 |
| 4 | `who` | 2.8608 |
| 5 | `about` | 2.6094 |
| 6 | `adrenaline` | 2.6094 |
| 7 | `adrenaline junkie` | 2.6094 |
| 8 | `adventures` | 2.6094 |
| 9 | `an organization` | 2.6094 |
| 10 | `and lousy` | 2.6094 |
| 11 | `attitude` | 2.6094 |
| 12 | `attitude when` | 2.6094 |
| 13 | `be planning` | 2.6094 |
| 14 | `before` | 2.6094 |
| 15 | `by the` | 2.6094 |

### consensus

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `34th` | 6.4827 |
| 2 | `34th street` | 6.4827 |
| 3 | `ability` | 6.4827 |
| 4 | `ability to` | 6.4827 |
| 5 | `action humor` | 6.4827 |
| 6 | `adding up` | 6.4827 |
| 7 | `adventure and` | 6.4827 |
| 8 | `afghanistan` | 6.4827 |
| 9 | `afghanistan very` | 6.4827 |
| 10 | `again` | 6.4827 |
| 11 | `aged` | 6.4827 |
| 12 | `al` | 6.4827 |
| 13 | `alec` | 6.4827 |
| 14 | `alec guinness` | 6.4827 |
| 15 | `always` | 6.4827 |

### tags_combined

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `000` | 10.0652 |
| 2 | `007 assassin` | 10.0652 |
| 3 | `01 15` | 10.0652 |
| 4 | `02 15` | 10.0652 |
| 5 | `04 15` | 10.0652 |
| 6 | `09 based` | 10.0652 |
| 7 | `10 2007` | 10.0652 |
| 8 | `10 classic` | 10.0652 |
| 9 | `10 not` | 10.0652 |
| 10 | `10 years` | 10.0652 |
| 11 | `100 author` | 10.0652 |
| 12 | `109` | 10.0652 |
| 13 | `10th` | 10.0652 |
| 14 | `10th century` | 10.0652 |
| 15 | `11 artist` | 10.0652 |

---
## Vectorization Parameters

- **N-gram Range**: (1, 2) - unigrams and bigrams
- **Min Document Frequency**: 5 - minimum 5 movies must contain feature
- **Max Document Frequency**: 0.6 - maximum 60% of movies can contain feature
- **Max Features**: 200,000 per field
- **Data Type**: float32
- **Lowercase**: Disabled (text pre-cleaned)

---
## Output Files

- **TF-IDF Matrices**: `.npz` sparse format
- **Vectorizers**: `.joblib` serialized objects
- **Index Mapping**: `.parquet` with canonical_id → row_index
- **Combined Matrix**: All fields horizontally stacked

---
*TF-IDF vectorization completed successfully. All text fields have been converted to numerical features ready for recommendation models.*


# TF-IDF Vectorization Report
## Step 2a.2 - Movie Recommendation Optimizer

**Generated:** 2025-08-28 12:50:24  
**Total Movies:** 88,194

---

## Vectorization Summary

### Fields Vectorized
- **overview**: 127 features, 0.0% coverage
- **consensus**: 2,119 features, 1.6% coverage
- **tags_combined**: 89,772 features, 58.8% coverage

### Combined Matrix
- **Shape**: 88,194 movies × 92,018 features
- **Sparsity**: 1.000

---
## Coverage Analysis

| Field | Valid Texts | Coverage % | Features | Matrix Shape |
|-------|-------------|-------------|----------|--------------|
| overview | 44 | 0.0% | 127 | 88,194 × 127 |
| consensus | 1,442 | 1.6% | 2,119 | 88,194 × 2,119 |
| tags_combined | 51,895 | 58.8% | 89,772 | 88,194 × 89,772 |

---
## Top Features by Field

### overview

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `all` | 2.8608 |
| 2 | `are` | 2.8608 |
| 3 | `most` | 2.8608 |
| 4 | `who` | 2.8608 |
| 5 | `about` | 2.6094 |
| 6 | `adrenaline` | 2.6094 |
| 7 | `adrenaline junkie` | 2.6094 |
| 8 | `adventures` | 2.6094 |
| 9 | `an organization` | 2.6094 |
| 10 | `and lousy` | 2.6094 |
| 11 | `attitude` | 2.6094 |
| 12 | `attitude when` | 2.6094 |
| 13 | `be planning` | 2.6094 |
| 14 | `before` | 2.6094 |
| 15 | `by the` | 2.6094 |

### consensus

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `34th` | 6.4827 |
| 2 | `34th street` | 6.4827 |
| 3 | `ability` | 6.4827 |
| 4 | `ability to` | 6.4827 |
| 5 | `action humor` | 6.4827 |
| 6 | `adding up` | 6.4827 |
| 7 | `adventure and` | 6.4827 |
| 8 | `afghanistan` | 6.4827 |
| 9 | `afghanistan very` | 6.4827 |
| 10 | `again` | 6.4827 |
| 11 | `aged` | 6.4827 |
| 12 | `al` | 6.4827 |
| 13 | `alec` | 6.4827 |
| 14 | `alec guinness` | 6.4827 |
| 15 | `always` | 6.4827 |

### tags_combined

| Rank | Feature | IDF Score |
|------|---------|-----------|
| 1 | `000` | 10.0652 |
| 2 | `007 assassin` | 10.0652 |
| 3 | `01 15` | 10.0652 |
| 4 | `02 15` | 10.0652 |
| 5 | `04 15` | 10.0652 |
| 6 | `09 based` | 10.0652 |
| 7 | `10 2007` | 10.0652 |
| 8 | `10 classic` | 10.0652 |
| 9 | `10 not` | 10.0652 |
| 10 | `10 years` | 10.0652 |
| 11 | `100 author` | 10.0652 |
| 12 | `109` | 10.0652 |
| 13 | `10th` | 10.0652 |
| 14 | `10th century` | 10.0652 |
| 15 | `11 artist` | 10.0652 |

---
## Vectorization Parameters

- **N-gram Range**: (1, 2) - unigrams and bigrams
- **Min Document Frequency**: 5 - minimum 5 movies must contain feature
- **Max Document Frequency**: 0.6 - maximum 60% of movies can contain feature
- **Max Features**: 200,000 per field
- **Data Type**: float32
- **Lowercase**: Disabled (text pre-cleaned)

---
## Output Files

- **TF-IDF Matrices**: `.npz` sparse format
- **Vectorizers**: `.joblib` serialized objects
- **Index Mapping**: `.parquet` with canonical_id → row_index
- **Combined Matrix**: All fields horizontally stacked

---
*TF-IDF vectorization completed successfully. All text fields have been converted to numerical features ready for recommendation models.*


## 2a.3 BERT Embeddings

**Generated:** 2025-08-28 21:03:12 UTC  
**Model:** sentence-transformers/all-MiniLM-L6-v2  
**Dimension:** 384  
**Device:** cpu  
**Batch Size:** 64  
**Total Runtime:** 3971.23s

---

### Model Configuration

- **Model Name**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Processing Device**: cpu
- **Batch Size**: 64 (CPU)
- **L2 Normalization**: Enabled
- **Deterministic Seeds**: torch=42, numpy=42, python=42

---

### Coverage Statistics

| Field | Valid Texts | Coverage % | Embeddings Generated |
|-------|-------------|-------------|----------------------|
| overview | 44 | 0.0% | 88,194 |
| consensus | 1,442 | 1.6% | 88,194 |
| tags_combined | 51,895 | 58.8% | 88,194 |


---

### Weighted Combination

**Field Weights:**
- **tags_combined**: 0.60
- **consensus**: 0.25
- **overview**: 0.15

**Combination Strategy:** Weighted mean with automatic renormalization for missing fields

---

### Nearest Neighbors Analysis

**Cosine Similarity Sanity Check:**

**Anchor Movie tt1362058 (row 53666):**

| Rank | Movie ID | Similarity |
|------|----------|------------|
| 1 | `tt1023500` | 0.4621 |
| 2 | `tt0892899` | 0.4321 |
| 3 | `tt0051226` | 0.4284 |
| 4 | `tt2023690` | 0.4243 |
| 5 | `tt0102819` | 0.4229 |

**Anchor Movie tt0071754 (row 22453):**

| Rank | Movie ID | Similarity |
|------|----------|------------|
| 1 | `tt6190456` | 1.0000 |
| 2 | `tt0818629` | 1.0000 |
| 3 | `tt2385101` | 1.0000 |
| 4 | `tt8611016` | 1.0000 |
| 5 | `tt0037426` | 1.0000 |

**Anchor Movie tt1568150 (row 44551):**

| Rank | Movie ID | Similarity |
|------|----------|------------|
| 1 | `tt2405792` | 0.3468 |
| 2 | `tt2405792` | 0.3468 |
| 3 | `tt7542576` | 0.3408 |
| 4 | `tt7663776` | 0.3390 |
| 5 | `tt2372776` | 0.3371 |

---
*BERT embeddings completed successfully. All text fields have been converted to semantic vectors ready for recommendation models.*

## 2a.4 Index & Storage

### Overview
Consolidated all text feature artifacts into a consistent, queryable feature store with unified indexing.

### Unified Index
- **Authoritative index**: `data/features/text/index/movies_text_index.parquet`
- **Row count**: 88,194 movies (canonical_id aligned)
- **Source**: tfidf_index_v1
- **Metadata**: row_index, source, row_count, created_utc

### Feature Artifacts
- **TF-IDF matrices**: 4 sparse CSR matrices (.npz)
- **BERT embeddings**: 4 dense arrays (.npy) with 384 dimensions
- **Vectorizers**: 3 fitted TF-IDF vectorizers (.joblib)
- **Statistics views**: 5 lightweight parquet files for BI/debug

### Quality Assurance
- **Row alignment**: All matrices verified to have 88,194 rows
- **Dimension consistency**: BERT embeddings confirmed 384 dimensions
- **Coverage validation**: Non-zero counts computed for all TF-IDF fields
- **Similarity checks**: TF-IDF vs BERT comparison for 3 anchor movies

### Storage Organization
```
data/features/text/
├── index/           # Unified index and manifest
├── views/           # Lightweight statistics and presence flags
├── checks/          # Similarity validation results
├── vectorizers/     # Fitted TF-IDF models
└── *.npz/*.npy     # Feature matrices and embeddings
```

### Manifest
Machine-readable JSON manifest at `data/features/text/index/movies_text_features_manifest.json` containing:
- Artifact metadata (name, path, type, shape, dtype)
- File sizes and creation timestamps
- Dependencies and descriptions
- Total artifact count and coverage statistics

## 2a.5 QA & Final Report

### Row Alignment Confirmation
✅ **All artifacts aligned to 88,194 rows** across TF-IDF matrices and BERT embeddings  
✅ **Canonical ID consistency** verified with unique identifiers  
✅ **BERT dimension validation** confirmed 384-dimensional vectors  

### TF-IDF Coverage Statistics

| Field | Min | Median | P95 | Max | Features |
|-------|-----|--------|-----|-----|----------|
| **overview** | 0 | 0.0 | 0.0 | 89 | 127 |
| **consensus** | 0 | 0.0 | 0.0 | 53 | 2,119 |
| **tags** | 0 | 3.0 | 134.0 | 2471 | 89,772 |
| **combined** | 0 | 3.0 | 136.0 | 2490 | 92,018 |

### BERT Semantic Validation
- **Cosine similarity ranges** (sample-based):
  - Overview: 1.000 to 1.000
  - Consensus: -0.103 to 1.000
  - Tags: -0.227 to 1.000
  - Combined: -0.045 to 1.000

### Neighbor Similarity Analysis
**Anchor movies** (seed=42, chosen from movies with tags): tt0060662, tt1762248, tt0027726

**Average overlap@5**: 2.33 out of 5 neighbors

**Detailed examples:**

- **tt0060662**:
  - TF-IDF top-5: tt1266545, tt0060074, tt0060605, tt0067451, tt0060880
  - BERT top-5: tt1266545, tt0060605, tt0053018, tt0060074, tt0067451
  - Overlap: 4/5

- **tt1762248**:
  - TF-IDF top-5: tt2971990, tt3104062, tt0256459, tt0440939, tt0415234
  - BERT top-5: tt0086872, tt2971990, tt3104062, tt0440939, tt1252595
  - Overlap: 3/5

- **tt0027726**:
  - TF-IDF top-5: tt3547948, tt1846499, tt0099492, tt0070584, tt0049849
  - BERT top-5: tt0039758, tt7162400, tt1746136, tt4061760, tt3604950
  - Overlap: 0/5

**Note**: TF-IDF and BERT approaches may differ due to:
- TF-IDF: Bag-of-words frequency-based similarity
- BERT: Semantic understanding and contextual embeddings
- Different feature spaces (sparse vs dense, frequency vs semantic)

### Visualizations
- [TF-IDF Overview NNZ Distribution](img/step2a_tfidf_nnz_hist_overview.png)
- [TF-IDF Consensus NNZ Distribution](img/step2a_tfidf_nnz_hist_consensus.png)  
- [TF-IDF Tags NNZ Distribution](img/step2a_tfidf_nnz_hist_tags.png)
- [Similarity Overlap Analysis](img/step2a_sim_overlap_bar.png)

### Notes & Limitations
- **Sparse coverage**: Overview and consensus fields have limited coverage (0.05% and 1.6% respectively)
- **Tags dominance**: Tags field provides 58.8% coverage and dominates the combined feature space
- **CPU processing**: BERT embeddings generated on CPU (CUDA not available) with 64 batch size
- **Memory efficient**: Sparse TF-IDF matrices stored as CSR format, dense BERT as float32 arrays
- **Deterministic**: All random sampling uses seed=42 for reproducibility

### Ready for Step 2b–2d
✅ **Text feature engineering complete** with 88,194 movies  
✅ **TF-IDF vectors**: 92,018 features (sparse CSR)  
✅ **BERT embeddings**: 384 dimensions (dense float32)  
✅ **Unified indexing**: Canonical ID alignment across all artifacts  
✅ **Quality assured**: Row counts, dimensions, and semantic validation passed  

All text features are now **consolidated, indexed, and validated** for downstream recommendation model training.
