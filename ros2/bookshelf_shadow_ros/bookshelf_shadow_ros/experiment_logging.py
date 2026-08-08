"""Pure helpers for reproducible bookshelf experiment logging."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def sha256_file(path) -> str | None:
    path = str(path or "").strip()
    if not path:
        return None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_snapshot(repository_path) -> dict:
    repository = Path(repository_path).expanduser().resolve()
    output = {"repository": str(repository)}
    commands = {
        "commit": ["git", "-C", str(repository), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(repository), "branch", "--show-current"],
        "status_short": ["git", "-C", str(repository), "status", "--short"],
    }
    for name, command in commands.items():
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            output[name] = result.stdout.strip() if result.returncode == 0 else None
            if result.returncode != 0:
                output[f"{name}_error"] = result.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            output[name] = None
            output[f"{name}_error"] = str(error)
    output["clean"] = output.get("status_short") == ""
    return output
