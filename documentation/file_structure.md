/api
├── services/
│   ├── __init__.py
│   ├── organization_service.py
│   ├── semester_service.py
│   ├── applicant_service.py
│   ├── matching_service.py
│   └── feedback_service.py
│
├── middleware/
│   ├── __init__.py
│   ├── rate_limit.py
│   ├── logging_middleware.py
│   └── correlation_id.py
│
├── routers/
│   ├── __init__.py
│   ├── auth_router.py
│   ├── organization_router.py
│   ├── semester_router.py
│   ├── applicant_router.py
│   ├── matching_router.py
│   ├── feedback_router.py
│   └── model_router.py
│
└── dependencies.py (Dependency Injection)

/database/repositories/
├── __init__.py
├── base_repository.py
├── user_repository.py
├── organization_repository.py
├── semester_repository.py
├── applicant_repository.py
├── match_group_repository.py
└── feedback_repository.py

tests/
├── test_services.py          # Main unit tests
│   ├── TestOrganizationService (10 tests)
│   ├── TestSemesterService (10 tests)
│   ├── TestApplicantService (9 tests)
│   ├── TestMatchingService (8 tests)
│   ├── TestFeedbackService (5 tests)
│   └── TestServiceIntegration (1 test)
│
├── test_edge_cases.py        # Edge cases & error handling
│   ├── TestOrganizationEdgeCases (4 tests)
│   ├── TestSemesterEdgeCases (5 tests)
│   ├── TestApplicantEdgeCases (4 tests)
│   ├── TestMatchingEdgeCases (3 tests)
│   ├── TestRaceConditions (2 tests)
│   └── TestErrorRecovery (2 tests)
│
└── pytest.ini 