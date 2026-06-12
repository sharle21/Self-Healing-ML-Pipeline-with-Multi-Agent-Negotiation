from self_healing_pipeline.gateway import Incident, IncidentType


def test_fingerprint_deterministic_for_same_payload() -> None:
    p1 = {"feature": "AGE", "z": 3.1}
    p2 = {"z": 3.1, "feature": "AGE"}
    a = Incident(tenant_id="standard", type=IncidentType.DRIFT, payload=p1)
    b = Incident(tenant_id="standard", type=IncidentType.DRIFT, payload=p2)
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_with_payload() -> None:
    a = Incident(tenant_id="standard", type=IncidentType.DRIFT, payload={"feature": "AGE"})
    b = Incident(tenant_id="standard", type=IncidentType.DRIFT, payload={"feature": "LIMIT_BAL"})
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_scoped_by_tenant_and_type() -> None:
    payload = {"feature": "AGE"}
    base = Incident(tenant_id="standard", type=IncidentType.DRIFT, payload=payload)
    other_tenant = Incident(tenant_id="enterprise", type=IncidentType.DRIFT, payload=payload)
    other_type = Incident(tenant_id="standard", type=IncidentType.DATA_QUALITY, payload=payload)
    assert base.fingerprint() != other_tenant.fingerprint()
    assert base.fingerprint() != other_type.fingerprint()


def test_ids_are_unique() -> None:
    a = Incident(tenant_id="standard", type=IncidentType.DRIFT, payload={})
    b = Incident(tenant_id="standard", type=IncidentType.DRIFT, payload={})
    assert a.id != b.id
