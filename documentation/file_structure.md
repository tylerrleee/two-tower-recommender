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