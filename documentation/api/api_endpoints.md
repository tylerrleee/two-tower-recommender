# API Endpoint Design (Complete REST API)

### Endpoint Structure
```
/api/v1  (API Versioning)
│
├── /auth
│   ├── POST   /login           - Authenticate user, get JWT
│   ├── POST   /register        - Register new coordinator (admin-only)
│   ├── GET    /me              - Get current user info
│   └── POST   /refresh         - Refresh expired JWT token
│
├── /organizations
│   ├── GET    /                - List all orgs (admin-only)
│   ├── POST   /                - Create new org (admin-only)
│   ├── GET    /:org_id         - Get org details
│   ├── PATCH  /:org_id         - Update org settings
│   └── GET    /:org_id/members - List org members
│
├── /semesters
│   ├── GET    /                          - List semesters for user's org
│   ├── POST   /                          - Create new semester
│   ├── GET    /:semester_id              - Get semester details
│   ├── PATCH  /:semester_id              - Update semester (name, dates)
│   ├── DELETE /:semester_id              - Delete semester (soft delete)
│   └── GET    /:semester_id/applicants   - Get all applicants for semester
│
├── /applicants
│   ├── POST   /upload          - Upload CSV of applicants
│   ├── GET    /:semester_id    - List applicants by semester
│   ├── POST   /                - Create single applicant (manual entry)
│   ├── PATCH  /:applicant_id   - Update applicant profile
│   └── DELETE /:applicant_id   - Delete applicant
│
├── /matching
│   ├── POST   /batch                      - Trigger batch matching (async job)
│   ├── GET    /jobs/:job_id/status        - Poll job status
│   ├── GET    /results/:semester_id       - Get match results
│   ├── GET    /results/:semester_id/export - Download CSV/JSON
│   └── POST   /results/:semester_id/approve - Approve and finalize matches
│
├── /matches 
│   ├── GET    /:match_id               - Get details of a specific match pair
│   ├── PATCH  /:match_id               - Manual Override (Swap mentor/mentee)
│   └── POST   /:match_id/feedback      - Submit feedback (rating/comment) for a match
│
├── /feedback <-- NEW RESOURCE
│   ├── GET    /summary/:semester_id    - Get aggregate satisfaction stats
│   ├── GET    /export                  - Download labeled dataset for retraining
│   └── POST   /batch                   - Upload bulk feedback (e.g., end-of-semester survey CSV)
│
├── /model
│   ├── GET    /info            - Get loaded model metadata
│   ├── GET    /versions        - List available model versions
│   ├── POST   /reload          - Reload model (admin-only)
│   ├── POST   /train                   - Trigger fine-tuning on collected feedback
│   └── GET    /train/:job_id/status    - Poll training job status
│
└── /health
    └── GET    /                - Health check (no auth required)
```