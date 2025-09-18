# Movie Recommendation Optimizer Makefile

# Phase 1: Schema & Types
clean-phase1:
	python -m scripts.cleaning.phase1_schema_types

# Phase 2: ID Resolution & Deduping
clean-phase2:
	python -m scripts.cleaning.phase2_id_resolution
# Optional: Run with fuzzy matching (slower but more comprehensive)
# clean-phase2-fuzzy:
# 	python -m scripts.cleaning.phase2_id_resolution --max-fuzzy 5000 --fuzzy-threshold 92 --year-tolerance 1

# Step 2a: Text Feature Engineering
text-tfidf:
	python scripts/text/build_tfidf.py

text-bert:
	python scripts/text/build_bert.py \
		--cleaned data/features/text/checkpoints/cleaned_text.parquet \
		--index data/features/text/movies_text_tfidf_index.parquet \
		--outdir data/features/text/

text-index:
	python scripts/text/index_text_features.py \
		--index data/features/text/movies_text_tfidf_index.parquet \
		--tfidf_over data/features/text/movies_text_tfidf_overview.npz \
		--tfidf_cons data/features/text/movies_text_tfidf_consensus.npz \
		--tfidf_tags data/features/text/movies_text_tfidf_tags_combined.npz \
		--tfidf_comb data/features/text/movies_text_tfidf_combined.npz \
		--bert_over data/features/text/movies_text_bert_overview.npy \
		--bert_cons data/features/text/movies_text_bert_consensus.npy \
		--bert_tags data/features/text/movies_text_bert_tags_combined.npy \
		--bert_comb data/features/text/movies_text_bert_combined.npy \
		--bert_meta data/features/text/movies_text_bert_meta.parquet \
		--outdir data/features/text/

text-qa:
	python scripts/text/qa_text_features.py \
		--index data/features/text/movies_text_tfidf_index.parquet \
		--tfidf_over data/features/text/movies_text_tfidf_overview.npz \
		--tfidf_cons data/features/text/movies_text_tfidf_consensus.npz \
		--tfidf_tags data/features/text/movies_text_tfidf_tags_combined.npz \
		--tfidf_comb data/features/text/movies_text_tfidf_combined.npz \
		--bert_over data/features/text/movies_text_bert_overview.npy \
		--bert_cons data/features/text/movies_text_bert_consensus.npy \
		--bert_tags data/features/text/movies_text_bert_tags_combined.npy \
		--bert_comb data/features/text/movies_text_bert_combined.npy \
		--outdir data/features/text/

# Step 2d: Platform Features
platform-encode:
	python scripts/platform/phase2d_provider_encoding.py

platform-qa:
	python scripts/platform/phase2d_index_qa.py

# Step 3a: Composite Feature Assembly
cbf-assemble:
	python scripts/composite/step3a1_feature_matrix_assembly.py

# Step 3a.2: Cosine + kNN
cbf-knn:
	python scripts/similarity/step3a2_similarity_computation.py

# Step 3a.3: QA and validation
cbf-qa:
	python scripts/qa/step3a3_qa_spot_checks.py

# Step 3a.4: Final report (placeholder)
cbf-report:
	@echo "Step 3a.4: Final report - Not yet implemented"
	@echo "Input: All Step 3a artifacts"
	@echo "Output: Comprehensive report"

# Default target
all: clean-phase1 clean-phase2

.PHONY: clean-phase1 clean-phase2 text-tfidf text-bert text-index text-qa platform-encode cbf-assemble cbf-knn cbf-qa cbf-report all
