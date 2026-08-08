from pathlib import Path

from bookshelf_shadow_ros.experiment_logging import git_snapshot, sha256_file


def test_sha256_file_is_deterministic_and_missing_is_none(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"bookshelf")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64
    assert sha256_file(tmp_path / "missing") is None
    assert sha256_file("") is None


def test_git_snapshot_fails_closed_for_non_repository(tmp_path):
    snapshot = git_snapshot(Path(tmp_path))
    assert snapshot["repository"] == str(tmp_path.resolve())
    assert snapshot["commit"] is None
    assert not snapshot["clean"]
