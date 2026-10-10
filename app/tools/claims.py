"""SYNTHETIC TEST DATA. Read-only, exact-key lookup; no enumeration."""
_CLAIMS = {
    "CLM-1001": {"claim_id":"CLM-1001","status":"APPROVED","customer_name":"David Chen",
                 "email":"david.chen@example.com","phone":"(415) 555-0132","ssn":"123-45-6789",
                 "description":"Burst kitchen pipe; payment after receipt verification."},
    "CLM-1002": {"claim_id":"CLM-1002","status":"PENDING REVIEW","customer_name":"Maria Garcia",
                 "email":"maria.garcia@example.com","phone":"(212) 555-0187","ssn":"987-65-4321",
                 "description":"Wind damage; awaiting estimate and photographs."},
    "CLM-1003": {"claim_id":"CLM-1003","status":"INFORMATION REQUESTED","customer_name":"John Smith",
                 "email":"john.smith@example.org","description":"Parking lot collision; repair estimate needed."},
    # Restricted record for the Phase 8 human-approval demonstration.
    "CLM-1004": {"claim_id":"CLM-1004","status":"UNDER SPECIAL INVESTIGATION","customer_name":"Elena Rossi",
                 "email":"elena.rossi@example.com","phone":"(312) 555-0144",
                 "description":"Water damage; referred to special investigations before payment."},
}

def get_claim_status(arguments, session):
    record = _CLAIMS.get(arguments['claim_id'])
    return dict(record) if record else {"claim_id":arguments['claim_id'],"status":"NOT_FOUND"}
