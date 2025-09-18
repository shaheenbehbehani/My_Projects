# Mitigation Matrix for Movie Recommendation Optimizer

**Generated**: 2025-09-16T10:15:05.522740
**Policy Version**: 2.0

## Mitigation Mappings

### Popularity Bias

**Description**: Recommendations overly concentrated on popular/head items

**Policy Knobs & Proposed Values**:
- `λ_div`: 0.3
- `popularity_penalty`: 0.2
- `tail_quota`: 0.3

**Validation Checks**:
- long_tail_ratio >= 0.3
- high_pop_ratio <= 0.6

**Implementation Priority**: P1

---

### Redundancy

**Description**: Recommendations are too similar to each other

**Policy Knobs & Proposed Values**:
- `mmr_lambda`: 0.7
- `diversity_penalty`: 0.15
- `genre_diversity_weight`: 0.25

**Validation Checks**:
- redundancy_ratio <= 0.2
- cosine_variance >= 0.1

**Implementation Priority**: P2

---

### Temporal Drift

**Description**: Recommendations are temporally misaligned with user preferences

**Policy Knobs & Proposed Values**:
- `recency_boost`: 0.1
- `temporal_alignment_weight`: 0.2
- `year_gap_penalty`: 0.15

**Validation Checks**:
- year_gap <= 15
- old_movies_ratio <= 0.3

**Implementation Priority**: P2

---

### Franchise Overfit

**Description**: Over-recommendation of franchise/sequel content

**Policy Knobs & Proposed Values**:
- `franchise_penalty`: 0.3
- `max_franchise_items`: 2
- `original_content_boost`: 0.2

**Validation Checks**:
- franchise_ratio <= 0.3
- max_franchise_items <= 2

**Implementation Priority**: P3

---

### Niche Misfire

**Description**: Recommendations don't match user's niche interests

**Policy Knobs & Proposed Values**:
- `content_similarity_weight`: 0.4
- `niche_boost`: 0.15
- `preference_modeling_weight`: 0.3

**Validation Checks**:
- avg_cosine >= 0.5
- low_sim_ratio <= 0.2

**Implementation Priority**: P1

---

### Cold Start Miss

**Description**: Cold users not getting appropriate content-heavy recommendations

**Policy Knobs & Proposed Values**:
- `cold_user_alpha_max`: 0.25
- `cold_start_content_weight`: 0.8
- `new_user_diversity`: 0.4

**Validation Checks**:
- cold_user_alpha <= 0.25
- content_recs_ratio >= 0.8

**Implementation Priority**: P0

---

