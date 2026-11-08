"""Capacity accounting must prevent duplicate jobs and avoid idle worker slots."""

from assembly_policy.compute.autoscale import additional_jobs, shape, starts_within


def config(**kw):
    return dict(
        enabled=True, base_job_ids=["1"], worker_slots=512, max_chimera_cpus=1000, cloud_workers=0, **kw
    )


def test_pending_capacity_counts_and_slot_limit_wins():
    q = [dict(id="1", state="RUNNING", cpus=400), dict(id="2", state="PENDING", cpus=100)]
    c = config()
    c["worker_slots"] = 260
    assert additional_jobs(c, q, dict(jobs=["2"])) == 2
    c["cloud_workers"] = 10
    assert additional_jobs(c, q, dict(jobs=["2"])) == 0


def test_ambiguous_submit_blocks_retry_and_disabled_run_cannot_submit():
    assert additional_jobs(config(), [], dict(jobs=[], uncertain_submission={})) == 0
    assert additional_jobs(config(), [], dict(jobs=[], uncertain_submission={"time": 1})) == 0
    c = config()
    c["enabled"] = False
    assert additional_jobs(c, [], dict(jobs=[])) == 0


def test_own_host_exclusion_and_faithful_cpu_shape():
    a = shape(100, "CPU1,GPU2")
    assert "--exclude=CPU1,GPU2" in a
    assert "--ntasks=50" in a and "--cpus-per-task=2" in a
    assert "--requeue" in a and "--time=02:00:00" in a


def test_only_positive_fresh_scheduler_projection_can_scale():
    import datetime as dt

    now = dt.datetime(2026, 11, 7, 10).timestamp()
    row = dict(returncode=0, stdout="", stderr="Job 1 to start at 2026-11-07T10:02:00")
    assert starts_within(row, now)
    assert not starts_within(row, now - 600)
    row["returncode"] = 1
    assert not starts_within(row, now)


def test_observation_only_sweep_never_submits_and_throttles(tmp_path, monkeypatch):
    from assembly_policy.compute import autoscale

    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "3077705")
    monkeypatch.setenv("USER", "test")

    def cmd(argv, **_):
        calls.append(argv)
        if argv[0] == "squeue":
            return dict(returncode=0, stdout="10|RUNNING|400|CPU01\n", stderr="")
        if argv[0] == "scontrol":
            return dict(returncode=0, stdout="TotalNodes=29", stderr="")
        assert "--test-only" in argv
        assert "--exclude=CPU01" in argv
        return dict(returncode=1, stdout="", stderr="Unavailable")

    monkeypatch.setattr(autoscale, "command", cmd)
    assert autoscale.cycle(tmp_path)["action"] == "observe_only_no_registered_run"
    n = len(calls)
    assert autoscale.cycle(tmp_path)["cached"]
    assert len(calls) == n


def test_one_threaded_process_per_ten_cpus_counts_slots_and_pending():
    c = config(cpus_per_worker=10)
    c["worker_slots"] = 100
    q = [dict(id="1", state="RUNNING", cpus=350), dict(id="2", state="PENDING", cpus=640)]
    assert additional_jobs(c, q, dict(jobs=["2"])) == 1
    c["worker_slots"] = 99
    assert additional_jobs(c, q, dict(jobs=["2"])) == 0
