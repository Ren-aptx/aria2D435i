from scripts.run_humanego_downstream import clear_invalidated_results


def test_clear_invalidated_results_preserves_completed_upstream_stage(tmp_path):
    preprocess = tmp_path / "preprocess"
    preprocess.mkdir()
    names = (
        "kptsselector_results.json",
        "cotracker_results.json",
        "camtriangulator_results.json",
    )
    for name in names:
        (preprocess / name).write_text("stale")

    removed = clear_invalidated_results(tmp_path, "cotracker")

    assert [path.name for path in removed] == [
        "cotracker_results.json", "camtriangulator_results.json"
    ]
    assert (preprocess / "kptsselector_results.json").is_file()
    assert not (preprocess / "cotracker_results.json").exists()
    assert not (preprocess / "camtriangulator_results.json").exists()
