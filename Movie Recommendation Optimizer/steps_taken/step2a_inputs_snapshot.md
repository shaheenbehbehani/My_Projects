# Step 2a.0 Input Audit & Setup Report
## Movie Recommendation Optimizer - Text Fields Inventory

**Generated:** $(date)  
**Total Movie Universe:** 87,601 movies  
**Objective:** Prepare and validate all text fields required for feature engineering before any cleaning or embeddings

---

## Executive Summary

This audit identifies and validates all available text fields across the movie recommendation datasets. The analysis covers 7 primary datasets with varying levels of text coverage, from basic metadata to rich narrative content.

**Key Findings:**
- **High Coverage:** Title fields (100%), Genre fields (100%), Basic metadata (100%)
- **Medium Coverage:** Overview/synopsis (TMDB: 100%, RT Top Movies: 99%)
- **Rich Content:** Reviews (RT: 1.4M+, IMDB: 50K labeled, 46K modified)
- **Tags:** MovieLens tags (2M+ user-generated tags)

---

## Dataset Inventory

### 1. Master Movies Dataset
**File:** `data/normalized/movies_master.parquet`  
**Row Count:** 87,601  
**Key Text Columns:**
- `title` - Movie title (87,601 non-null, 100% coverage)
- `title_norm` - Normalized title (87,601 non-null, 100% coverage)
- `genres_norm` - Normalized genres list (87,601 non-null, 100% coverage)
- `genres_str` - Genre string (87,601 non-null, 100% coverage)

**Sample Rows:**
```
canonical_id: tt0114709
title: Toy Story
title_norm: toy story
genres_norm: [comedy, adventure, animation]
genres_str: adventure|animation|comedy

canonical_id: tt0039758
title: The Red Stallion
title_norm: the red stallion
genres_norm: [western, drama, family]
genres_str: drama|family|western
```

---

### 2. IMDB Dataset
**File:** `data/normalized/imdb/title_basics.parquet`  
**Row Count:** 11,856,706  
**Key Text Columns:**
- `primaryTitle` - Primary title (11,856,706 non-null, 100% coverage)
- `originalTitle` - Original title (11,856,706 non-null, 100% coverage)
- `genres` - Genre list (11,856,706 non-null, 100% coverage)

**File:** `data/normalized/imdb/title_crew.parquet`  
**Row Count:** 11,858,387  
**Key Text Columns:**
- `directors` - Director IDs (11,858,387 non-null, 100% coverage)
- `writers` - Writer IDs (11,858,387 non-null, 100% coverage)

**Sample Rows:**
```
primaryTitle: Carmencita
originalTitle: Carmencita
genres: Documentary,Short

primaryTitle: Le clown et ses chiens
originalTitle: Le clown et ses chiens
genres: Animation,Short
```

---

### 3. MovieLens Dataset
**File:** `data/normalized/movielens/movies.parquet`  
**Row Count:** 87,585  
**Key Text Columns:**
- `title` - Movie title with year (87,585 non-null, 100% coverage)
- `genres` - Genre string (87,585 non-null, 100% coverage)

**File:** `data/normalized/movielens/tags.parquet`  
**Row Count:** 2,000,072  
**Key Text Columns:**
- `tag` - User-generated tag (2,000,072 non-null, 100% coverage)

**Sample Rows:**
```
title: Toy Story (1995)
genres: Adventure|Animation|Children|Comedy|Fantasy

tag: Kevin Kline
tag: misogyny
tag: acrophobia
tag: music
tag: weird
```

---

### 4. Rotten Tomatoes Dataset
**File:** `data/normalized/rottentomatoes/movies.parquet`  
**Row Count:** 143,258  
**Key Text Columns:**
- `title` - Movie title (143,258 non-null, 100% coverage)
- `genre` - Genre list (143,258 non-null, 100% coverage)
- `director` - Director names (143,258 non-null, 100% coverage)
- `writer` - Writer names (143,258 non-null, 100% coverage)

**File:** `data/normalized/rottentomatoes/top_movies.parquet`  
**Row Count:** 1,610  
**Key Text Columns:**
- `title` - Movie title (1,610 non-null, 100% coverage)
- `consensus` - Critical consensus (1,593 non-null, 99% coverage)
- `synopsis` - Movie synopsis (available in schema)

**File:** `data/normalized/rottentomatoes/reviews.parquet`  
**Row Count:** 1,444,963  
**Key Text Columns:**
- `reviewText` - Review content (1,444,963 non-null, 100% coverage)

**Sample Rows:**
```
title: Space Zombie Bingo!
genre: Comedy, Horror, Sci-fi
director: George Ormrod
writer: George Ormrod,John Sabotta

title: Black Panther
consensus: Black Panther elevates superhero cinema to thrilling new heights while telling one of the MCU's most absorbing stories -- and introducing some of its most fully realized characters.

reviewText: Timed to be just long enough for most youngsters' brief attention spans -- and it's packed with plenty of interesting activity, both on land and under the water.
```

---

### 5. TMDB Dataset
**File:** `data/normalized/tmdb/movies.parquet`  
**Row Count:** 600  
**Key Text Columns:**
- `title` - Movie title (600 non-null, 100% coverage)
- `overview` - Movie overview (600 non-null, 100% coverage)
- `genres` - Genre information (600 non-null, 100% coverage)

**Sample Rows:**
```
title: War of the Worlds
overview: Will Radford is a top analyst for Homeland Security who tracks potential threats through a mass surveillance program, until one day an attack by an unknown entity leads him to question whether the government is hiding something from him... and from the rest of the world.

title: Mission: Impossible - The Final Reckoning
overview: Ethan Hunt and team continue their search for the terrifying AI known as the Entity — which has infiltrated intelligence networks all over the globe — with the world's governments and a mysterious ghost from Hunt's past on their trail.
```

---

### 6. Labeled Review Datasets

#### 6.1 IMDB Reviews (Modified)
**File:** `IMDB datasets/Movies_Reviews_modified_version1.csv`  
**Row Count:** 46,173  
**Key Text Columns:**
- `Reviews` - Review content (46,173 non-null, 100% coverage)
- `emotion` - Emotion label (46,173 non-null, 100% coverage)

#### 6.2 IMDB Reviews (Standard)
**File:** `IMDB datasets/IMDB Dataset.csv`  
**Row Count:** 50,000  
**Key Text Columns:**
- `review` - Review content (50,000 non-null, 100% coverage)
- `sentiment` - Sentiment label (50,000 non-null, 100% coverage)

**Sample Rows:**
```
Reviews: It had some laughs, but overall the motivation of the characters was incomprehensible...
emotion: anticipation

review: One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked...
sentiment: positive
```

---

## Coverage Analysis

### Text Field Coverage by Dataset

| Dataset | Movies | Title | Genre | Overview/Synopsis | Reviews | Tags |
|---------|--------|-------|-------|-------------------|---------|------|
| Master Movies | 87,601 | 100% | 100% | N/A | N/A | N/A |
| IMDB | 11.8M+ | 100% | 100% | N/A | N/A | N/A |
| MovieLens | 87,585 | 100% | 100% | N/A | N/A | 100% |
| Rotten Tomatoes | 143,258 | 100% | 100% | 99%* | 100% | N/A |
| TMDB | 600 | 100% | 100% | 100% | N/A | N/A |
| Labeled Reviews | 96,173 | N/A | N/A | N/A | 100% | N/A |

*Consensus field in top_movies subset

### Coverage Relative to Movie Universe (87,601)

| Text Field Type | Total Available | Coverage % | Notes |
|-----------------|----------------|-------------|-------|
| **Titles** | 87,601 | 100% | Master dataset + normalized variants |
| **Genres** | 87,601 | 100% | Multiple genre representations |
| **Overview/Synopsis** | 2,210 | 2.5% | TMDB (600) + RT Top Movies (1,610) |
| **Reviews** | 1,540,963 | N/A | Rich review corpus across platforms |
| **Tags** | 2,000,072 | N/A | User-generated MovieLens tags |
| **Crew** | 87,601 | 100% | Director/writer information available |

---

## Data Quality Observations

### Strengths
1. **Complete Coverage:** Title and genre fields have 100% coverage across the master dataset
2. **Rich Metadata:** Multiple genre representations (normalized lists, strings, original formats)
3. **Abundant Reviews:** Over 1.5M review texts available for sentiment analysis
4. **User Tags:** 2M+ MovieLens tags provide community-driven categorization
5. **Multiple Sources:** Diverse text content from IMDB, RT, TMDB, and MovieLens

### Limitations
1. **Overview Coverage:** Only 2.5% of movies have detailed overview/synopsis text
2. **Review Distribution:** Reviews are not evenly distributed across all movies
3. **Text Length Variation:** Significant variation in text field lengths and quality
4. **Language Mix:** Some datasets contain non-English content

### Missing Text Fields
- **Plot summaries** for majority of movies (97.5% missing)
- **Character descriptions** or cast details
- **Production notes** or behind-the-scenes information
- **User reviews** for many movies in the master dataset

---

## Recommendations for Feature Engineering

### Phase 1: Core Text Fields
1. **Titles:** Use `title` and `title_norm` for exact and fuzzy matching
2. **Genres:** Leverage multiple genre representations for enhanced categorization
3. **Basic Metadata:** Director, writer, and production information

### Phase 2: Rich Content Fields
1. **Reviews:** Sentiment analysis and keyword extraction from 1.5M+ reviews
2. **Tags:** Community-driven categorization from 2M+ user tags
3. **Overview/Synopsis:** Detailed text analysis for available movies

### Phase 3: Derived Features
1. **Text Embeddings:** Generate embeddings for all available text fields
2. **Sentiment Scores:** Aggregate review sentiment by movie
3. **Keyword Extraction:** Extract key terms from reviews and synopses
4. **Language Detection:** Identify and handle multilingual content

---

## Next Steps

**Step 2a.1:** Text Cleaning and Preprocessing
- Standardize text formats and encodings
- Handle missing values and edge cases
- Prepare text for embedding generation

**Step 2a.2:** Text Vectorization
- Generate embeddings for all text fields
- Implement dimensionality reduction
- Create feature matrices for recommendation models

---

## File Locations

- **Master Dataset:** `data/normalized/movies_master.parquet`
- **IMDB Data:** `data/normalized/imdb/`
- **MovieLens Data:** `data/normalized/movielens/`
- **Rotten Tomatoes Data:** `data/normalized/rottentomatoes/`
- **TMDB Data:** `data/normalized/tmdb/`
- **Labeled Reviews:** `IMDB datasets/`

---

*This audit confirms that sufficient text data is available to proceed with feature engineering. The combination of structured metadata, rich review content, and user-generated tags provides a solid foundation for building text-based recommendation features.*

























