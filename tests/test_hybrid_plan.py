import hashlib
import json

import pytest

from assembly_policy.compute.hybrid import plan, validate_qualification


def pool(provider="modal"):
    return dict(
        provider=provider,
        hosts=1,
        workers=4,
        threads=2,
        seconds=3600,
        memory_mib=16384,
        max_cost_usd=5,
        estimated_hourly_usd=1,
        region="us-east",
    )


def test_mixed_fleet_has_one_shared_ticket_and_fixed_slot_limit():
    c = dict(pools=[pool(), pool("sail")], credits_usd=dict(modal=1500, sail=1500))
    rows = plan(c, dict(tls_ca="pinned", n_conns=8), "/private/ticket.json")
    assert [r["provider"] for r in rows] == ["modal", "sail"]
    assert all("/private/ticket.json" in r["command"] for r in rows)
    with pytest.raises(ValueError, match="slots"):
        plan(c, dict(tls_ca="pinned", n_conns=7), "/private/ticket.json")


def test_credit_budget_is_aggregate_per_provider():
    c = dict(pools=[pool(), pool()], credits_usd=dict(modal=8, sail=1500))
    with pytest.raises(ValueError, match="credits"):
        plan(c, dict(tls_ca="pinned", n_conns=10), "/private/ticket.json")


def test_cloud_ticket_requires_tls():
    with pytest.raises(ValueError, match="TLS"):
        plan(dict(pools=[]), dict(n_conns=8), "/private/ticket.json")


def test_main_run_requires_qualified_current_artifacts(tmp_path):
    source = tmp_path / "model.xml"
    source.write_text("frozen model")
    receipt = dict(status="blocked", checks={})
    manifest = tmp_path / "qualification.json"
    config = dict(qualification_manifest="qualification.json")
    manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="not qualified"):
        validate_qualification(tmp_path / "config.json", config)
    receipt.update(status="qualified", checks=dict.fromkeys([
        "physical_reset_admission", "current_cad_worker_parity", "full_ppo_resume",
        "throughput_budget", "trainer_backend",
    ], True), artifacts=[dict(path="model.xml", sha256=hashlib.sha256(source.read_bytes()).hexdigest())])
    manifest.write_text(json.dumps(receipt))
    validate_qualification(tmp_path / "config.json", config)
    source.write_text("later CAD edit")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_qualification(tmp_path / "config.json", config)
