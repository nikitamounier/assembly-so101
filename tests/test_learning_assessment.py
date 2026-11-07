from scripts.cloud.assess_learning import summarize


def result(phase, success, safety, error=0.01):
    return dict(
        phase=phase,
        episodes=[
            dict(success=success, safety=safety, timeout=not success and not safety, best_error_m=error)
        ],
    )


def test_weight_updates_or_error_only_do_not_count_as_task_success():
    report = summarize([result("initial", False, False, 0.02), result("final", False, False, 0.001)])
    assert report["task_improvement"] == "not_demonstrated_on_this_cohort"


def test_success_gain_is_reported_on_its_measured_cohort():
    report = summarize([result("initial", False, False), result("final", True, False)])
    assert report["task_improvement"] == "demonstrated_on_this_cohort"


def test_more_success_with_more_safety_stops_is_not_a_clean_improvement():
    rows = [
        result("initial", False, False),
        result("initial", False, False),
        result("final", True, False),
        result("final", False, True),
    ]
    assert summarize(rows)["task_improvement"] == "not_demonstrated_on_this_cohort"
