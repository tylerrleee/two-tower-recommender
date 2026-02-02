
### 1. Data Loading & Preprocessing
1. load_csv_data()
   ├─ Reads CSV file
   ├─ Renames columns using config.RENAME_MAP
   ├─ Splits into df_mentors and df_mentees 
   └─ Validates required columns exist

2. engineer_features()
   ├─ Creates FeatureEngineer with config fields
   ├─ Fits on entire dataset (mentors + mentees)
   │  ├─ Applies OneHotEncoding to categorical fields - e.g. (0,0,1,0..)
   │  ├─ Applies StandardScaler to numeric fields - range of [0,1]
   │  └─ Builds ColumnTransformer -- consolidate into a single object
   └─ Transforms mentor_data and mentee_data separately
      └─ Output: profile_text + meta_features for each object

3. generate_embeddings()
   ├─ Loads S-BERT model 
   ├─ Encodes profile_text → text embeddings (default: 384 dim)
   ├─ Concatenates text embeddings + meta features
   └─ Output: mentor_embedding, mentee_embedding
      Example shape: (n_mentors, 384 + meta_dim)

### 2. Model Initialization
```
4. initialize_model()
   ├─ Calculates meta_feature_dim = total_dim - 384
   ├─ Creates TwoTowerModel
   │  ├─ Input: embedding_dim (384) + meta_feature_dim
   │  ├─ Architecture: [256, 128, 64] hidden layers
   │  ├─ Separate towers for mentor & mentee
   │  └─ Output: 64-dimensional learned embeddings
   └─ Moves model to CPU (current: MacOS Sillicon)
```
###  3: Training Setup** (in train_model())
```
5. Bootstrap Positive Pairs
   With use_multi_positive=True, k_positives=3:
   
   ├─ Computes similarity matrix (mentors × mentees)
   ├─ Uses Hungarian algorithm for optimal matching
   ├─ For each mentor:
   │  ├─ Assigns best match as pos_pairs[i, 0]
   │  └─ Finds k-1 nearest neighbors as pos_pairs[i, 1:k]
   │
   └─ Output: pos_pairs shape (n_mentors, 3)
      Example: pos_pairs[0] = [342, 891, 567]
              -> Mentor 0's top 3 mentees

6. Prepare Diversity Features
   With loss_type = 'margin':
   ├─ Creates dummy diversity features (zeros)
   └─ Not used in margin loss, but required for dataset

7. Hard Negative Pool
   With use_hard_negatives=True:
   ├─ Collects all unused mentees (not in any top-k)
   ├─ Creates hard_negative_pool from their embeddings
   └─ Used for harder training examples

8. Train/Val Split
   ├─ Splits at MENTOR level (not pairs!)
   ├─ Ensures validation tests on unseen mentors
   └─ Prevents overfitting to specific mentors
```

### ** 4: Dataset & DataLoader Creation**
```
9. Create MentorMenteeDataset
   ├─ Wraps embeddings as PyTorch tensors
   ├─ In multi-positive mode:
   │  └─ __getitem__(i) randomly samples 1 of k positives
   │     -> Provides gradient diversity across epochs
   │
   └─ Optionally includes hard negatives per sample

10. Create DataLoaders
    ├─ Train: batch_size=32, shuffle=True
    └─ Val: batch_size=32, shuffle=False
```

### 5: Training Loop (20 epochs)
```
11. For each epoch:
    
    A. TRAINING
       For each batch (32 mentor-mentee pairs):
       
       ├─ Forward pass through TwoTowerModel
       │  ├─ mentor_feat -> mentor_tower -> mentor_emb (64-dim)
       │  ├─ mentee_feat -> mentee_tower -> mentee_emb (64-dim)
       │  └─ Both embeddings are L2-normalized from embeddings.py (S-BERT)
       │
       ├─ Compute PairwiseMarginLoss
       │  ├─ Similarity matrix: mentor_emb @ mentee_emb.T
       │  ├─ Positive similarities: sim[i, positive_pairs[i]]
       │  ├─ Loss = max(0, margin - pos_sim + neg_sim)
       │  └─ Goal: pos_sim > neg_sim + margin (0.2)
       │
       ├─ Backward pass
       │  ├─ Compute gradients
       │  ├─ Clip gradients (max_norm=1.0)
       │  └─ Update weights with Adam optimizer
       │
       └─ Track loss
    
    B. VALIDATION
       ├─ Same forward pass (no gradient computation)
       ├─ Compute validation loss
       └─ Check for improvement
    
    C. EARLY STOPPING
       ├─ If val_loss improves: save checkpoint
       ├─ If no improvement for X epochs: stop training
       └─ Load best checkpoint at end
```

### Phase 6: Post-Training
```
12. generate_mentor_embeddings()
    ├─ model.eval() → freeze weights
    ├─ Forward pass on ALL mentors & mentees
    ├─ Extract learned 64-dim embeddings
    └─ Store as mentor_embeddings_learned, mentee_embeddings_learned

13. match_groups(use_faiss=False)
    ├─ Uses learned embeddings (not raw S-BERT)
    ├─ GroupMatcher.find_best_groups_base()
    │  ├─ Duplicates each mentor (2 slots per mentor)
    │  ├─ Computes similarity: learned_mentor @ learned_mentee.T
    │  ├─ Hungarian algorithm assigns 2 mentees per mentor
    │  └─ Maximizes global compatibility
    │
    └─ Output: groups dict {mentor_idx: {mentees, scores}}


## Front-end
