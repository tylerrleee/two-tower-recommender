
# Mentor-Mentee Matching System
## Deep Learning-Based Big/Little Pairing for University Programs

This system uses a Two-Tower Neural Network architecture with contrastive learning to match mentors ("Bigs") with mentees ("Littles") in university mentorship programs. By combining semantic text understanding (S-BERT), structured profile features, and objective optimization, the system generates mentor-to-mentee groups that balance compatibility and diversity.


## TODO
- generate_embeddings_with_cache(Data Ingestion Scaffold (CSV → MongoDB)
- Applied CRUD Operations
-- CREATE: Insert New Mentor Application
-- READ: Find Unmatched Applicants for a Semester
-- UPDATE: Save Computed S-BERT Embedding
-- DELETE: Remove Test Application
-Integration Testing Script : Integration tests for MongoDB adapter with ML pipeline.
- scripts/init_db.py  # Initialize MongoDB with indexes and test data.


## Reason for this Project

Traditional approaches to match 100-1000 applications rely on manual review or simple keyword matching. This system aims to:

- Learns semantic similarity from profile text using pre-trained language models (S-BERT)
- Bootstraps training labels using S-BERT embeddings to create intelligent pseudo-pairs
- Optimizes globally using the Hungarian algorithm to maximize overall match quality
- Encourages diversity within mentee pairs to prevent echo chambers (groups can't grow or explore each other since they are very similar to each other)
- Scales efficiently to handle 1,000+ applicants in seconds using FAISS acceleration
- Reduces coordinator burden from 40+ hours to < 2 hours per matching cycle

## Similar Works

Similar works for Big-Little in fraternities and sororities have been done many times. However, the downside to the Greek life point of view is the scale of matching, where popular matching algorithms like Genetic & Stable Marriage relies on labeled preferences for all candidates for each respective role. Hence, having 100+ applicants is unrealistic to have an applicant rank everyone, when they have never met, know or have the time to rank everyone themselves. 

Stable Marriage Problem (CMU approach):

- Requires every applicant to rank every other applicant
- For 300 mentors + 600 mentees: 360,000 rankings needed 
- Unrealistic when applicants haven't met each other

Genetic Algorithms (Hatch-a-Match):

- Still requires preference initialization
- Computationally expensive (population evolution)
- Hard to tune for different organizations

This project's Content-Based Filtering/Approach:

- Zero manual rankings required 
- Learns compatibility from profile features automatically
- Transfers across organizations with different questions

<a href="https://github.com/haley/hatchamatch?tab=readme-ov-file"> Hatch-a-Match - Genetic Algorithm</a>

<a href="https://www.math.cmu.edu/users/math/af1p/Teaching/OR2/Projects/P44/E.pdf"> Big-Little Pairings at Carnegie Mellon Univeristy - Stable Marriage Problem & Greedy</a>

-----
## System Architecture

TODO: Diagram

## Core Components

### 1. **Feature Engineering** (`features.py`)

Transform raw survey data into compatible formats for Machine Learning:

#### **Text Processing** (free response) fields -> 1 combined profile)

Example Corpus:

```python
profile_text_fields = [
    'free_time',              # "Love hiking and photography"
    'hobbies',                # "Playing guitar, cooking"
    'self_description',       # "Introverted but friendly"
    'dislikes',               # "Loud environments"
    'talk_for_hours_about',   # "Philosophy and movies"
    'friday_night',           # "Reading or game night"
    'additional_info'         # Free-form notes
]

# Output: "Love hiking and photography. Playing guitar, cooking. Introverted..."
```
** Goal: ** Semantic concatenation preserves context across multiple fields, allowing S-BERT to understand holistic personality. 


**Key Innovation:** Semantic concatenation preserves context across multiple fields, allowing S-BERT to understand holistic personality.

**Issue & Future Plan:** A single concatenated array may present signals that are overlooked, which it struggles to capture complex indenitifes or diverse interests. Pan et al., 2025, highlights that "a single dense embedding struggles to disentangle these heterogeneous signals, constraining the model’s expressiveness," so it is necessary to explore multiple embeddings in text processing.

#### **Categorical Encoding** (One-Hot)
```python
major → [0, 1, 0, 0, 0, 0]  # Computer Science | 6 possible majors [F T F F F F]
sleep_schedule → [1, 0]     # Early bird | [True, False]
```

#### **Numerical Normalization** (Z-score)
```python
# Before normalization
extroversion: 4 (on 1-5 scale)
study_frequency: 3
gym_frequency: 2

# After StandardScaler (mean=0, std=1)
extroversion: 0.52
study_frequency: 0.01
gym_frequency: -0.48
```

#### **Diversity Features**
```python
def compute_diversity_features(self, df):
    # Extroversion balance (favor moderate vs extremes)
    extro_complement = 1.0 - np.abs((df['extroversion'] - 3) / 2)
    
    # Study-social balance
    study_social_balance = np.abs(df['study_frequency'] - df['gym_frequency'])
    
    return np.column_stack([extro_complement, study_social_balance])
```

**Result:** 400D feature vector (384 text + ~16 meta) per applicant

---

### 2. **Embedding Generation** (`embedding.py`)

Uses [S-BERT](https://www.sbert.net/) to create semantic embeddings:

```python
# S-BERT Model: 'all-MiniLM-L6-v2'
"I love outdoor activities" → [0.23, -0.45, 0.67, ..., 0.12]  # 384D
"I enjoy hiking and camping" → [0.25, -0.43, 0.69, ..., 0.09]  # 384D

cosine_similarity = 0.92  # From 0 to 1, semantically similar!
```

**Why S-BERT over TF-IDF?**
- Pre-trained on 1B+ sentence pairs (transfer learning)
- Captures semantic meaning: "studying abroad" is similar to "international experience"
- Contextual understanding vs. simple keyword matching 
- - Keyword matching often occurs in manual matching as applicants increase, so context and semantic understanding is the key criteria for this project.



**Feature Combination:**
```python
text_embedding (384D) + meta_features (16D) = 400D input vector

# Example:
combined = np.concatenate([
    sbert_vector,        # [0.23, -0.45, ..., 0.12] (384D)
    onehot_major,        # [0, 1, 0, 0, 0, 0] (6D) 
    onehot_sleep,        # [1, 0] (2D)
    normalized_likert    # [0.52, 0.01, -0.48, ...] (6D)
], axis=1)

```

---

### 3. **Bootstrap Positive Pairs** (`bootstrap_positive_pairs.py`)

**The Cold Start Problem:**

Traditional supervised learning requires labeled training data (e.g., "Mentor A matches well with Mentee B"). However, for new mentorship programs, we have no historical match data to learn from.

Why don't we use previous match cycles for transfer learning? 
- For student organizations, age generations and lack of standardize matching criteria, past applications' survey questions and structures are different from present day. 
- Different match coordinator means human bias prone to keyword matching, preferential match or experience bias with certain applicants. 

**Solution: Pseudo-Labeling**

- Instead of random or arbitrary pairings, we use S-BERT embeddings to bootstrap training labels.
- This means that we are labeling semantically similar combined embeddings as positive pairs. 

**Issue:** 

- This does not entirely solve the Cold Start Problem, where survey data is not entirely reprensentative of one's preference, or predictive of their experience with their match
- On a human level, some applicant's response are not with their full effort, or with the most effective representation of themselves. 

**Example Bootstrap Output:**
```
Bootstrapped pairs with avg similarity: 0.734

Top 5 pairs:
  Mentor 0 (CS, hiking) ↔ Mentee 42 (Data Sci, outdoors) | Score: 0.89
  Mentor 1 (Bio, reading) ↔ Mentee 17 (Chem, books) | Score: 0.85
  Mentor 2 (Econ, sports) ↔ Mentee 33 (Business, gym) | Score: 0.82
  ...
```

### 4. **Two-Tower Model** (`model.py`)

Dual encoder architecture that learns to refine the bootstrap embeddings:

# TODO; Diagram

**What the Model Learns:**

Starting from bootstrap pairs, the model learns:

1. **Feature importance weights**
   - Text semantics: 60%
   - Major compatibility: 20%
   - Personality traits: 15%
   - Study/social habits: 5%

2. **Example: Non-linear interactions**
   - CS mentor + Bio mentee might work IF shared interest in research
   - Extrovert mentor + Introvert mentee balanced by similar hobbies

3. **Refined embedding space**
   - Pushes good matches closer (0.73 → 0.89)
   - Separates bad matches further (0.45 → 0.12)

**Training Progress Example:**
```
Epoch 1:
  Bootstrap avg: 0.73
  Positive sim: 0.68 (worse - model is random)
  Negative sim: 0.51
  
Epoch 10:
  Positive sim: 0.82 (better than bootstrap!)
  Negative sim: 0.28
  Gap: 0.54
  
Epoch 50:
  Positive sim: 0.91 (much better!)
  Negative sim: 0.15
  Gap: 0.76 
```

**Why Two Towers?**
1. **Separate Encoders:** Mentors and mentees can have different learned representations
2. **Efficient Inference:** Pre-compute embeddings offline, then fast similarity search O(log n) on user interfance (TODO)

**Model Design Choices:**

1. Batch Normalization: Stabilize training to enable higher learning rate and faster convergence

2. Dropout: Prevent overfitting, especially with small applicant pools (100-1000)

3. L2 Normalization: Embeddings -> unit vectors -> cosine similarity to scale invariant matching on bounded scores [-1,1]

4. Learnable Bias: Calibrate score distribution to adapt to feature specific simiarlity ranges

### 5. **Loss Functions** (`loss.py`, `pairwise_margin_loss.py`)

# TODO

### 6. **Matching Algorithm** (`matcher.py`)

### 7. **Training Pipeline** (`train.py`)

# TODO

## Alteranatives | Recommendation/Matching System Design Choices

While researching about recommendation systems, I found many designs and resources out there:

#TODO: expand on design chocie

1. Collaborative Filtering

2. Term Frequency-Inverse Document Frequency + Cosine Similarity

3. Graph Neural Network

4. Stable Marriage

5. Genetic Algothithm 

6. Two Tower Neural Network


## Complete Pipeline 

# TODO

## Installation & Usage

# TODO 


# References / Resources used (pending)

Van Rossum, G., Warsaw, B., & Coghlan, N. (2001). PEP 8 – Style Guide for Python Code. Python Software Foundation. 
https://peps.python.org/pep-0008/
​

Python Software Foundation. (2026). unittest — Unit testing framework (Python 3.14.2 documentation). 
https://docs.python.org/3/library/unittest.html
​

Google Developers. (n.d.). Recommendation systems. In Machine Learning Crash Course. 
https://developers.google.com/machine-learning/recommendation
​

Totten, J. (2023). TensorFlow deep retrieval using Two-Tower architecture. Google Cloud Blog. 
https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture
​

Fuxi-MME Authors. (2025). Revisiting scalable sequential recommendation with Multi-Embedding Approach and Mixture-of-Experts. arXiv preprint. https://arxiv.org/html/2510.25285v1
​

Mlshark. (2023). InfoNCE explained in details and implementations. Medium. https://medium.com/@mlshark/infonce-explained-in-details-and-implementations-902f28199ce6
​
https://medium.com/@mingc.me/deploying-pytorch-model-to-production-with-fastapi-in-cuda-supported-docker-c161cca68bb8