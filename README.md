# Mentor-Mentee Matching System

A production-ready neural matching system that automatically pairs mentors with mentees in university mentorship programs using deep learning, semantic embeddings, and optimization algorithms.

## Overview

This system addresses the scalability challenge in mentor-mentee matching programs where manual review becomes infeasible beyond 300 applicants. By combining natural language processing (S-BERT), structured profile features, and global optimization (Hungarian algorithm), the system generates optimal 1:2 mentor-to-mentee groups that maximize compatibility while maintaining diversity.

## Transparency

Claude Code was used. 

**Key Features:**
- Zero manual preference ranking required
- Semantic understanding of free-text responses
- Multi-semester data persistence with MongoDB
- RESTful API for integration with existing systems
- Embedding caching for fast retraining
- Scalable to 1000+ applicants

## Problem Statement

Traditional matching approaches face critical limitations:

**Stable Marriage Problem (CMU Approach):**
- Requires every applicant to rank every other applicant
- For 300 mentors + 600 mentees: 360,000 rankings needed
- Unrealistic when applicants have never met

**Genetic Algorithms (Hatch-a-Match):**
- Requires preference initialization
- Computationally expensive population evolution
- Difficult hyperparameter tuning per organization

**Our Content-Based Approach:**
- Learns compatibility from profile features automatically
- No manual rankings or preferences needed
- Transfers across organizations with different survey structures

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MongoDB Atlas                     │
│  [applicants] [semesters] [match_groups]           │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │   DataAdapter     │
         │ Connection Manager│
         └─────────┬─────────┘
                   │
┌──────────────────┴───────────────────────────────┐
│             ML Pipeline (src/)                   │
│                                                  │
│  1. Feature Engineering    (features.py)        │
│     ├─ Text concatenation                       │
│     ├─ One-hot encoding                         │
│     └─ Numerical normalization                  │
│                                                  │
│  2. Embedding Generation   (embedding.py)       │
│     ├─ S-BERT (all-MiniLM-L6-v2)               │
│     └─ Feature combination                      │
│                                                  │
│  3. Bootstrap Pairs        (bootstrap_*.py)     │
│     ├─ Similarity matrix                        │
│     └─ Hungarian assignment                     │
│                                                  │
│  4. Two-Tower Training     (model.py, train.py) │
│     ├─ Dual encoder networks                    │
│     └─ Pairwise margin loss                     │
│                                                  │
│  5. Generate Embeddings    (model.py)           │
│     └─ 64-dim learned vectors                   │
│                                                  │
│  6. Final Matching         (matcher.py)         │
│     └─ Hungarian algorithm                      │
│                                                  │
│  7. Save Results           (saving_csv.py)      │
│     └─ CSV + JSON + MongoDB                     │
└──────────────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │   FastAPI (api/)  │
         │  - /health        │
         │  - /match/batch   │
         │  - /match/csv     │
         │  - /model/info    │
         └───────────────────┘
```

## Core Components

### 1. Feature Engineering (`src/features.py`)

Transforms raw survey responses into ML-ready features:

**Text Processing:**
```python
# Concatenate multiple free-response fields
profile_text = f"{bio}. {interests}. {goals}. {hobbies}"
# Output: "Love hiking and photography. Playing guitar, cooking. ..."
```

**Categorical Encoding:**
```python
major -> [0, 1, 0, 0, 0, 0]  # Computer Science (one-hot)
sleep_schedule -> [1, 0]      # Early bird
```

**Numerical Normalization:**
```python
# StandardScaler (mean=0, std=1)
extroversion: 4 -> 0.52
study_frequency: 3 -> 0.01
gym_frequency: 2 -> -0.48
```

**Output:** 400D feature vector (384 text + 16 meta) per applicant

### 2. Semantic Embeddings (`src/embedding.py`)

Uses S-BERT (Sentence-BERT) for contextual text understanding:

```python
model = SentenceTransformer('all-MiniLM-L6-v2')

"I love outdoor activities" -> [0.23, -0.45, 0.67, ..., 0.12]  # 384D
"I enjoy hiking and camping" -> [0.25, -0.43, 0.69, ..., 0.09]  # 384D

cosine_similarity = 0.92  # Semantically similar
```

**Advantages over TF-IDF:**
- Pre-trained on 1B+ sentence pairs (transfer learning)
- Captures semantic meaning: "studying abroad" ≈ "international experience"
- Contextual understanding vs keyword matching
- Robust to paraphrasing and synonyms

**Feature Combination:**
```python
combined = np.hstack([
    text_embedding,    # (n, 384)
    meta_features      # (n, 16)
])  # -> (n, 400)
```

### 3. Bootstrap Positive Pairs (`src/bootstrap_positive_pairs.py`)

**The Cold Start Problem:**

Traditional supervised learning requires labeled training data. However, new mentorship programs lack historical match quality labels.

**Solution: Pseudo-Labeling via S-BERT**

Instead of random pairings, use S-BERT embeddings to bootstrap initial training labels:

```python
# 1. Compute similarity matrix
similarity = mentor_embeddings @ mentee_embeddings.T

# 2. Hungarian algorithm for optimal 1:1 assignment
row_ind, col_ind = linear_sum_assignment(similarity, maximize=True)

# 3. Find k-1 nearest neighbors
for mentor_i in range(n_mentors):
    best_match = col_ind[mentor_i]
    sorted_indices = np.argsort(-similarity[mentor_i])
    candidates = [idx for idx in sorted_indices if idx != best_match]
    pos_pairs[mentor_i] = [best_match] + candidates[:k-1]
```

**Output:**
```
Bootstrapped 3 positives per mentor with avg similarity: 0.734

pos_pairs[0] = [342, 891, 567]  # Mentor 0's top-3 mentees
  342: Hungarian best match
  891, 567: 2nd and 3rd most similar
```

**Limitations:**
- Survey responses may not fully represent applicant preferences
- Assumes semantic similarity correlates with match quality
- Cannot account for unmeasured interpersonal chemistry

### 4. Two-Tower Neural Network (`src/model.py`)

Dual encoder architecture that refines bootstrap embeddings:

```
Input: (384 + meta_dim) combined embeddings

┌─────────────────────────────────┐
│      MENTOR TOWER               │
│  Linear(input_dim -> 256)       │
│  BatchNorm1d(256)               │
│  ReLU()                         │
│  Dropout(0.3)                   │
│          ↓                      │
│  Linear(256 -> 128)             │
│  BatchNorm1d(128)               │
│  ReLU()                         │
│  Dropout(0.3)                   │
│          ↓                      │
│  Linear(128 -> 64)              │
│  BatchNorm1d(64)                │
│  ReLU()                         │
│  Dropout(0.3)                   │
│          ↓                      │
│  L2 Normalize (64-dim)          │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│      MENTEE TOWER               │
│  [Same architecture]            │
└─────────────────────────────────┘

Output: 64-dim learned embeddings
```

**Model Design Rationale:**

1. **Batch Normalization:** Stabilizes training, enables higher learning rates
2. **Dropout (0.3):** Prevents overfitting with small datasets (100-1000 samples)
3. **L2 Normalization:** Scale-invariant cosine similarity, bounded scores [-1, 1]
4. **Separate Towers:** Mentors and mentees learn different representations

**Training Progress Example:**
```
Epoch 1:  Positive sim: 0.68 | Negative sim: 0.51 | Gap: 0.17
Epoch 10: Positive sim: 0.82 | Negative sim: 0.28 | Gap: 0.54
Epoch 20: Positive sim: 0.91 | Negative sim: 0.15 | Gap: 0.76
```

### 5. Pairwise Margin Loss (`src/pairwise_margin_loss.py`)

Contrastive learning objective that maximizes separation between positive and negative pairs:

```python
class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin=0.2, temperature=0.1):
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, mentor_emb, mentee_emb):
        # Similarity matrix
        sim = (mentor_emb @ mentee_emb.T) / temperature  # (B, B)
        
        # Positive similarities (diagonal)
        pos_sim = torch.diag(sim)  # (B,)
        
        # Loss: max(0, margin - pos_sim + neg_sim)
        loss = F.relu(margin - pos_sim.unsqueeze(1) + sim)
        
        # Mask diagonal (positives)
        mask = ~torch.eye(B, device=sim.device).bool()
        return loss[mask].mean()
```

**Goal:** pos_sim > neg_sim + margin (0.2)

**Training Configuration:**
- Optimizer: Adam (lr=1e-3)
- Epochs: 20
- Batch size: 32
- Gradient clipping: max_norm=1.0
- Early stopping: patience=5

### 6. Hungarian Matching (`src/matcher.py`)

Global optimization for final mentor-mentee group assignment:

```python
class GroupMatcher:
    def find_best_groups_base(self, mentor_emb, mentee_emb):
        # 1. Duplicate mentors (2 slots each)
        expanded_mentor = np.repeat(mentor_emb, 2, axis=0)
        
        # 2. Compute cost matrix
        cost_matrix = expanded_mentor @ mentee_emb.T
        
        # 3. Hungarian algorithm
        mentor_indices, mentee_indices = linear_sum_assignment(
            cost_matrix, maximize=True
        )
        
        # 4. Format results
        return self._format_results(mentor_indices, mentee_indices, cost_matrix)
```

**Complexity:**
- Base algorithm: O(n³) - exact optimal solution
- FAISS variant: O(k·n·log(m)) - approximate solution for large datasets

**Alternative: FAISS Acceleration (for 1000+ applicants)**
```python
def find_best_groups_faiss(self, mentor_emb, mentee_emb, top_k=10):
    # Build FAISS index for mentor embeddings
    index = faiss.IndexFlatIP(dim)
    index.add(mentor_emb)
    
    # Retrieve top-k mentors for each mentee
    distances, indices = index.search(mentee_emb, top_k)
    
    # Sparse cost matrix (only top-k candidates)
    sparse_cost_matrix = np.full((n_mentors * 2, n_mentees), -1e9)
    # ... populate sparse matrix ...
    
    # Hungarian on reduced search space
    return linear_sum_assignment(sparse_cost_matrix, maximize=True)
```

## MongoDB Schema

### Collection: `applicants`

```json
{
  "_id": "ObjectId(...)",
  "ufl_email": "jane.doe@ufl.edu",
  "full_name": "Jane Doe",
  "applications": [
    {
      "semester_id": "ObjectId(...)",
      "semester_code": "FALL_2026",
      "role": "mentor",
      "survey_responses": {
        "major": "Computer Science",
        "bio": "I love helping freshmen...",
        "interests": "AI, web dev",
        "extroversion": 4,
        "study_frequency": 5
      },
      "embeddings": {
        "sbert_384": [0.023, -0.145, ...],
        "learned_64": [0.412, -0.289, ...]
      },
      "matched_group_id": "ObjectId(...)"
    }
  ]
}
```

**Key Design:** Flexible `survey_responses` object allows changing questions per semester without schema migrations.

### Collection: `match_groups`

```json
{
  "_id": "ObjectId(...)",
  "semester_id": "ObjectId(...)",
  "mentor": {
    "applicant_id": "ObjectId(...)",
    "full_name": "Jane Doe",
    "major": "Computer Science"
  },
  "mentees": [
    {
      "applicant_id": "ObjectId(...)",
      "full_name": "John Smith",
      "compatibility_score": 0.873
    },
    {
      "applicant_id": "ObjectId(...)",
      "full_name": "Alice Wang",
      "compatibility_score": 0.821
    }
  ],
  "aggregate_score": 0.847,
  "status": "active"
}
```

## API Endpoints

### FastAPI Application (`api/main.py`)

**1. Health Check**
```http
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

**2. Batch Matching**
```http
POST /match/batch

Request:
{
  "applicants": [
    {
      "role": 0,  // 0=mentor, 1=mentee
      "name": "Jane Doe",
      "email": "jane@ufl.edu",
      "major": "Computer Science",
      "year": "Junior",
      "bio": "...",
      "interests": "...",
      "extroversion": 4
    }
  ],
  "use_faiss": false,
  "top_k": 10
}

Response:
{
  "status": "success",
  "total_groups": 150,
  "average_compatibility": 0.847,
  "groups": [...]
}
```

**3. CSV Upload**
```http
POST /match/csv
Content-Type: multipart/form-data

File: applications.csv
```

**4. Model Info**
```http
GET /model/info

Response:
{
  "model_loaded": true,
  "metadata": {
    "epoch": 15,
    "train_loss": 0.023,
    "val_loss": 0.031
  }
}
```

## Installation

### Prerequisites

- Python 3.9+
- MongoDB 4.4+ (local or Atlas)
- 8GB RAM minimum
- CUDA-compatible GPU (optional, for training)

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/mentor-mentee-matching.git
cd mentor-mentee-matching

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MONGODB_URI="mongodb://localhost:27017/"
export MONGODB_DATABASE="mentorship_matching"

# (Optional) Download pre-trained model
wget https://example.com/best_model.pt -O models/best_model.pt
```

### Dependencies (`requirements.txt`)

```
torch==2.0.0
transformers==4.30.0
sentence-transformers==2.2.2
fastapi==0.103.0
uvicorn==0.23.0
pymongo==4.5.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
python-dotenv==1.0.0
```

## Usage

### Training Pipeline

```python
from database.adapter import DataAdapter
from src.main import run_full_pipeline

# Initialize database adapter
adapter = DataAdapter()

# Fetch semester data
df_mentors, df_mentees = adapter.fetch_training_data(
    semester_id="67a1b2c3d4e5f6a7b8c9d0e2"
)

# Run end-to-end training + matching
results = run_full_pipeline(
    df_mentors=df_mentors,
    df_mentees=df_mentees,
    num_epochs=20,
    batch_size=32,
    k_positives=3
)

# Results saved to:
# - models/best_model.pt
# - output/matching_results_YYYYMMDD_HHMMSS/
```

### API Server

```bash
# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Test health endpoint
curl http://localhost:8000/health

# Run batch matching
curl -X POST http://localhost:8000/match/batch \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Output Files

After matching, results are saved to `output/matching_results_YYYYMMDD_HHMMSS/`:

**groups_summary.csv:**
```csv
Group ID,Mentor Name,Mentor Major,Num Mentees,Avg Compatibility
0,Jane Doe,Computer Science,2,0.847
1,John Smith,Biology,2,0.823
```

**detailed_matches.csv:**
```csv
Group ID,Mentor Name,Mentee Name,Individual Score
0,Jane Doe,Alice Wang,0.873
0,Jane Doe,Bob Lee,0.821
```

**results.json:**
```json
{
  "metadata": {
    "total_groups": 150,
    "avg_group_score": 0.847
  },
  "groups": [...]
}
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_api.py
pytest tests/test_features.py

# Run with coverage
pytest --cov=src --cov=api tests/
```

## Project Structure

```
two-tower-recommender/
├── api/                            # FastAPI Application Layer
│   ├── main.py                     # App factory and global settings
│   ├── models.py                   # Global Pydantic schemas/DTOs
│   ├── exceptions.py               # Custom HTTP & business logic exceptions
│   ├── inference.py                # Wrapper for model prediction
│   ├── dependencies.py             # FastAPI Dependency Injection (auth, DB sessions)
│   ├── middleware/                 # Cross-cutting concerns
│   │   ├── logging_middleware.py   # Request/Response logging
│   │   ├── rate_limit.py           # API throttling
│   │   └── correlation_id.py       # Distributed tracing
│   ├── routers/                    # Endpoint definitions
│   │   ├── auth_router.py          # User authentication
│   │   ├── organization_router.py  # Organization management
│   │   ├── semester_router.py      # Semester lifecycle
│   │   ├── applicant_router.py     # CSV uploads and applicant management
│   │   ├── matching_router.py      # Triggering matching jobs
│   │   ├── feedback_router.py      # Post-match surveys
│   │   └── model_router.py         # ML model metadata and retraining
│   └── services/                   # Business Logic Layer
│       ├── organization_service.py # Org logic and quotas
│       ├── semester_service.py     # Stats and status transitions
│       ├── applicant_service.py    # CSV validation and parsing
│       ├── matching_service.py     # Job status and matching orchestration
│       └── feedback_service.py     # Dataset export for ML retraining
│
├── database/                       # Infrastructure Layer
│   ├── connection.py               # MongoDB singleton connection manager
│   ├── adapter.py                  # PyMongo-to-Pandas/DataFrame adapter
│   └── repositories/               # Data Access Layer (CRUD)
│       ├── base_repository.py      # Common DB operations
│       ├── user_repository.py      # User profile management
│       ├── organization_repository.py
│       ├── semester_repository.py
│       ├── applicant_repository.py
│       ├── match_group_repository.py
│       └── feedback_repository.py
│
├── src/                            # Core ML Pipeline (Independent of API)
│   ├── main.py                     # E2E pipeline orchestration
│   ├── features.py                 # Text preprocessing and engineering
│   ├── embedding.py                # S-BERT embedding generation
│   ├── model.py                    # Two-tower PyTorch architecture
│   ├── train.py                    # Training loops and validation
│   ├── loss.py                     # Diversity/Alternative loss functions
│   ├── pairwise_margin_loss.py      # Ranking loss implementation
│   ├── bootstrap_positive_pairs.py # Multi-positive sampling logic
│   ├── matcher.py                  # Hungarian Algorithm for final pairing
│   └── saving_csv.py               # Logic for formatting outputs
│
├── scripts/                        # Maintenance and DevOps
│   ├── ingest_csv.py               # CSV-to-MongoDB migration tool
│   └── init_db.py                  # Schema/Index initialization
│
├── tests/                          # Automated Test Suite
│   ├── test_services.py            # Unit tests for all service layers
│   ├── test_edge_cases.py          # Validation and race condition tests
│   ├── test_api.py                 # Endpoint integration tests
│   ├── test_database.py            # Repository and adapter tests
│   ├── test_features.py            # ML feature engineering tests
│   └── test_embeddings.py          # S-BERT output verification
│
├── config/                         # Configuration Management
│   └── config.py                   # Environment variable management
│
├── models/                         # Model Artifacts
│   └── best_model.pt               # Trained model checkpoint
│
├── data/                           # Local Data Storage
│   └── applications.csv            # Sample/testing data
│
├── output/                         # Artifact Generation
│   └── matching_results_*/         # Result snapshots
│
├── .env                            # Sensitive environment variables
├── pytest.ini                      # Pytest configuration
└── requirements.txt                # Dependency list
```

## TODO

### Database Migration
- [ ] Implement `scripts/ingest_csv.py` for CSV → MongoDB migration
- [ ] Create `scripts/init_db.py` for database initialization with indexes

### Embedding Cache Enhancement
- [ ] Complete `generate_embeddings_with_cache()` in `src/embedding.py`
- [ ] Integrate MongoDB caching into training pipeline

### CRUD Operations
- [ ] CREATE: `database/crud.py` - Insert new mentor application
- [ ] READ: `database/crud.py` - Find unmatched applicants by semester
- [ ] UPDATE: `database/crud.py` - Save computed S-BERT embeddings
- [ ] DELETE: `database/crud.py` - Remove test applications

### Testing
- [ ] Complete `tests/test_embeddings.py` - S-BERT embedding tests
- [ ] Add `tests/test_mongodb_integration.py` - End-to-end pipeline tests
- [ ] Add performance benchmarks for large datasets (1000+ applicants)

### Documentation
- [ ] Add architecture diagrams (system flow, two-tower model, loss functions)
- [ ] Create API usage examples with curl/Python requests
- [ ] Document hyperparameter tuning guide

## Limitations and Future Work

### Current Limitations

1. **Cold Start Problem:** Bootstrap pairs assume semantic similarity correlates with match quality without ground truth validation
2. **Single Embedding:** Concatenated text may miss nuanced signals; multi-embedding approaches (Pan et al., 2025) could improve expressiveness
3. **Survey Dependency:** Model quality depends on applicant response quality and effort
4. **No Feedback Loop:** System lacks mechanism to incorporate post-match satisfaction data

### Future Enhancements

1. **Multi-Embedding Architecture:** Separate embeddings for bio, interests, and goals to preserve signal specificity
2. **Active Learning:** Incorporate coordinator feedback to refine model weights
3. **Explainability:** Generate human-readable match justifications (e.g., "Shared interest in AI research")
4. **Transfer Learning:** Pre-train on historical data from similar programs
5. **Dynamic Reweighting:** Adjust feature importance based on match success metrics

## Alternative Approaches Considered

### 1. Collaborative Filtering
**Pros:** Works well with historical preference data
**Cons:** Requires labeled past matches; cold start problem for new programs

### 2. TF-IDF + Cosine Similarity
**Pros:** Simple, interpretable
**Cons:** No semantic understanding; "studying abroad" ≠ "international experience"

### 3. Graph Neural Networks
**Pros:** Can model complex relationships
**Cons:** Requires graph structure; unclear how to construct initial edges

### 4. Stable Marriage Problem
**Pros:** Optimal with complete preference rankings
**Cons:** Unrealistic to collect 360,000 rankings for 300 mentors + 600 mentees

### 5. Genetic Algorithms
**Pros:** Flexible optimization framework
**Cons:** Requires fitness function design; computationally expensive

**Our Choice: Two-Tower Neural Network**
- Learns compatibility from unstructured data automatically
- Scales to 1000+ applicants with FAISS acceleration
- Transfers across organizations with different survey questions
- No manual preference ranking required

## References

Yi, X., Yang, J., Hong, L., et al. (2019). Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. *Proceedings of the 13th ACM Conference on Recommender Systems (RecSys '19)*, 269-277. https://doi.org/10.1145/3298689.3346996

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 815-823. https://doi.org/10.1109/CVPR.2015.7298682

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3982-3992. https://doi.org/10.18653/v1/D19-1410

Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly, 2*(1-2), 83-97. https://doi.org/10.1002/nav.3800020109

Pan, Z., Cai, F., Chen, W., et al. (2025). Revisiting Scalable Sequential Recommendation with Multi-Embedding Approach and Mixture-of-Experts. *arXiv preprint arXiv:2510.25285*. https://arxiv.org/abs/2510.25285

Van Rossum, G., Warsaw, B., & Coghlan, N. (2001). PEP 8 – Style Guide for Python Code. *Python Software Foundation*. https://peps.python.org/pep-0008/

Python Software Foundation. (2026). unittest – Unit testing framework (Python 3.14.2 documentation). https://docs.python.org/3/library/unittest.html

Google Developers. (n.d.). Recommendation systems. In *Machine Learning Crash Course*. https://developers.google.com/machine-learning/recommendation

Totten, J. (2023). TensorFlow deep retrieval using Two-Tower architecture. *Google Cloud Blog*. https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture

Mlshark. (2023). InfoNCE explained in details and implementations. *Medium*. https://medium.com/@mlshark/infonce-explained-in-details-and-implementations-902f28199ce6

**Related Projects:**
- Hatch-a-Match (Genetic Algorithm): https://github.com/haley/hatchamatch
- CMU Big-Little Pairings (Stable Marriage): https://www.math.cmu.edu/users/math/af1p/Teaching/OR2/Projects/P44/E.pdf

## License

MIT License - See LICENSE file for details

## Contributors

For technical questions or contributions, please open an issue or submit a pull request.

---

*Last Updated: January 30, 2026*
