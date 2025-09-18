# Netflix Movie Recommendation Optimizer

A comprehensive hybrid recommendation system that combines content-based filtering with collaborative filtering to deliver personalized movie recommendations at scale.

## 🎯 Project Overview

This project implements a production-ready movie recommendation system with the following key features:

- **Hybrid Architecture**: Combines TF-IDF/BERT embeddings with SVD matrix factorization
- **Adaptive Policy**: Bucket-gate policy with user cohort-based α blending
- **Comprehensive Evaluation**: 177 case studies across 4 user cohorts
- **Production Ready**: Robust error handling and data-driven policy optimization

## 📊 Key Performance Metrics

- **MAP@10**: 0.0066 (Hybrid Bucket-Gate)
- **Recall@10**: 0.0111
- **Coverage**: 0.247
- **Performance Lift**: +214.3% over Content-Based methods

## 🏗️ Project Structure

```
Movie Recommendation Optimizer/
├── visualizations/          # Step 5.1 professional visualizations
│   ├── scoreboard_k10.png
│   ├── lift_hybrid.png
│   ├── error_taxonomy.png
│   ├── policy_evolution.png
│   ├── system_pipeline.png
│   └── step5_visuals.log
├── report/                  # LinkedIn-ready reports
│   ├── step5_linkedin_report.pdf
│   ├── step5_linkedin_report.md
│   └── step5_linkedin_report.log
├── steps_taken/            # Step 1-4 progress documentation
│   ├── Step 1 Progress.docx
│   ├── Step 2 progress.docx
│   ├── Step 3 progress.docx
│   ├── Step 4 progress.docx
│   └── [markdown reports]
├── findings/               # Analysis results and insights
│   ├── step4_error_taxonomy.md
│   ├── step4_case_studies.md
│   ├── policy_step4_findings.md
│   └── step5_visuals.md
└── code/                   # Implementation code
    ├── scripts/
    │   ├── text/           # Text processing scripts
    │   ├── crew/           # Crew feature extraction
    │   ├── numeric/        # Numeric feature processing
    │   ├── eval/           # Evaluation scripts
    │   ├── serve/          # Serving scripts
    │   └── edgecases/      # Edge case testing
    ├── configs/            # Configuration files
    ├── Makefile
    └── requirements.txt
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   cd code
   pip install -r requirements.txt
   ```

2. **Run Feature Engineering**:
   ```bash
   make step2-features
   ```

3. **Train Models**:
   ```bash
   make step3-models
   ```

4. **Evaluate Performance**:
   ```bash
   make step4-eval
   ```

## 📈 Results Summary

### Performance Comparison
| Method | MAP@10 | Recall@10 | Coverage |
|--------|--------|-----------|----------|
| Content-Based | 0.0021 | 0.0113 | 0.709 |
| Collaborative Filtering | 0.00007 | 0.0007 | 0.010 |
| Hybrid α=1.0 | 0.0054 | 0.0111 | 0.247 |
| **Hybrid Bucket-Gate** | **0.0066** | **0.0111** | **0.247** |

### Error Analysis (177 Cases)
- **Redundancy**: 34.0% (161 cases)
- **Temporal Drift**: 27.4% (130 cases)
- **Stale Content**: 28.9% (137 cases)
- **Cold Start Miss**: 7.4% (35 cases)
- **Long-tail Starvation**: 2.3% (11 cases)

## 🔧 Technical Architecture

### Step 2: Feature Engineering
- TF-IDF vectors and BERT embeddings (87,601 × 384)
- Genre, crew, and standardized numeric features
- Comprehensive text processing and categorical encoding

### Step 3: Recommendation Models
- Content-based similarity with kNN neighbors
- Collaborative filtering via SVD (k=20, RMSE ~3.59)
- Hybrid pipeline with α blending and bucket-gate policy

### Step 4: Evaluation & Robustness
- Unified scoreboard (Recall@10, MAP@10, Coverage)
- Cohort analysis across cold/light/medium/heavy users
- Error taxonomy and case study validation

### Step 4.4: Policy Implementation
- Adaptive α selection based on user cohorts
- Long-tail quota and diversity enforcement
- Fallback strategies and emergency overrides

## 📋 Policy Evolution

### Policy v2.1 Improvements
- **Cold α**: 0.2 → 0.15 (tighter cold-start handling)
- **Tail Quota**: 0.0 → 0.3 (30% long-tail item guarantee)
- **MMR λ**: 0.0 → 0.7 (diversity enforcement)
- **Recency Boost**: 0.0 → 0.1 (temporal alignment)

## 🎨 Visualizations

The `visualizations/` folder contains 8 professional LinkedIn-ready charts:
- Performance scoreboard and lift analysis
- Error taxonomy and policy evolution
- System architecture and case study summaries
- All charts are 300 DPI PNG format

## 📄 Reports

The `report/` folder contains:
- **LinkedIn PDF Report**: Professional 3-page summary
- **Markdown Source**: Editable report source
- **Execution Logs**: Detailed generation logs

## 🔍 Key Findings

1. **Hybrid Bucket-Gate** achieves best overall performance
2. **214.3% performance lift** over Content-Based methods
3. **Error patterns** drive data-driven policy improvements
4. **Production ready** with comprehensive evaluation framework

## 🛠️ Dependencies

- Python 3.x
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- PyTorch (for BERT embeddings)
- Surprise (for collaborative filtering)

## 📝 License

This project is part of a portfolio demonstration and is available for educational purposes.

## 👨‍💻 Author

**Shaheen Behbehani**  
Portfolio Project | 2025

---

*Generated: 2025-01-27 | Reproducibility: Seed=42 | LinkedIn Ready*
