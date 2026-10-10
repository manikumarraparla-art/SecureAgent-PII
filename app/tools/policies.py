"""SYNTHETIC TEST DATA. Read-only, exact-key lookup."""
_POLICIES = {
    "POL-2001": {"policy_id":"POL-2001","policy_type":"Home","status":"ACTIVE",
                 "coverage":"Sudden accidental water damage; deductible 500 dollars.",
                 "customer_name":"David Chen","email":"david.chen@example.com"},
    "POL-2002": {"policy_id":"POL-2002","policy_type":"Home","status":"ACTIVE",
                 "coverage":"Windstorm roof damage; deductible 1000 dollars.",
                 "customer_name":"Maria Garcia","email":"maria.garcia@example.com"},
}

def get_policy_details(arguments, session):
    record = _POLICIES.get(arguments['policy_id'])
    return dict(record) if record else {"policy_id":arguments['policy_id'],"status":"NOT_FOUND"}
