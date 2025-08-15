
# 🚀 Founder Success Prediction Without Funding Data

## 📘 Overview

This project predicts startup founder success using non-funding data from Crunchbase. The model avoids circular logic by excluding funding-related features (e.g., raised capital, investor count) and instead evaluates founders on career trajectory, education, visibility, and execution signals.

- ✅ No funding features used  
- 📊 Built using clean Crunchbase data (People, Jobs, Degrees, Events, etc.)  
- 🤖 XGBoost classifier  
- 🏆 Success defined as: IPO or acquisition over $50M  
- 🔍 Features: Career acceleration, industry diversity, role progression, job tenure, visibility, executive presence, etc.  

---

## 🧾 Data Sources

- `people_descriptions.csv`  
- `jobs_clean.csv`  
- `degrees_clean.csv`  
- `organizations_clean.csv`  
- `events.csv`  
- `event_appearances.csv`  

---

## 🧹 Data Preparation

Each table was loaded using `polars` for speed and memory efficiency:

- Irrelevant or empty rows were dropped  
- Column types were standardized (`pl.Date`, `pl.Int64`, etc.)  
- Timestamps were parsed and missing values handled  
- Only verified founders were kept (`founder_verified = True`)  

---

## 🏗️ Feature Engineering

> No funding variables were used. Features were derived exclusively from career, education, event, and visibility metrics.

### Founder-Level Features

- `num_startups_founded`: Count of unique startups founded  
- `num_executive_roles`: Total roles with titles like CEO, CTO, etc.  
- `average_job_tenure`: Years per role  
- `career_acceleration`: Job switches per year  
- `industry_diversity`: Count of unique industries/sectors  
- `has_exited_company`: Whether a past job was at a company with IPO/acquisition  
- `years_of_experience`: From first job to last  
- `founder_age`: At first startup  

### Visibility Features

- `event_visibility_score`: Count of major appearances at Crunchbase-listed events  
- `speaker_ratio`: Ratio of speaker to attendee appearances  

### Education Features

- `num_degrees`: Count of completed degrees  
- `education_prestige_score`: Points for top-tier institutions (e.g., Stanford, MIT)  
- `executive_education`: MBA, JD, PhD flags  

---

## 🧠 Modeling Approach

- **Classifier**: XGBoost (`xgboost.XGBClassifier`)  
- **Train-Test Split**: 80/20  
- **Cross-validation**: 5-fold CV  
- **Hyperparameter Tuning**:
  - Learning rate  
  - Max depth  
  - Subsample ratio  
  - Colsample_bytree  

---

## ✅ Success Definition

A founder is marked as "successful" if **any of their startups:**

- IPO’d, or  
- Were acquired for **>$50 million**

This definition was implemented using Crunchbase’s `ipos` and `acquisitions` tables.

---

## 📈 Model Evaluation

| Metric       | Value     |
|--------------|-----------|
| ROC AUC      | 0.720     |
| F1 Score     | 0.079     |
| Accuracy     | 0.89+     |

> 🎯 Precision is low due to real-world class imbalance — most founders don’t reach exits. AUC remains strong, meaning the model ranks well despite skewed labels.

---

## 🔍 Feature Importance

Top Predictive Features (via XGBoost):

1. `num_startups_founded`  
2. `industry_diversity`  
3. `executive_presence_score`  
4. `career_acceleration`  
5. `event_visibility_score`  
6. `average_job_tenure`  
7. `education_prestige_score`  
8. `num_executive_roles`  
9. `has_exited_company`  
10. `speaker_ratio`
