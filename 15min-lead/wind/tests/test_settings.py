from __future__ import annotations

from pathlib import Path

from xinyang_wind15.settings import default_split_config_path


def test_default_split_config_prefers_local_on_windows(tmp_path: Path) -> None:
    splits_dir = tmp_path / "configs" / "splits"
    splits_dir.mkdir(parents=True)
    local_cfg = splits_dir / "xinyang_7_2_1.yaml"
    server_cfg = splits_dir / "xinyang_7_2_1_server.yaml"
    local_cfg.write_text("local\n", encoding="utf-8")
    server_cfg.write_text("server\n", encoding="utf-8")

    resolved = default_split_config_path(tmp_path, prefer_server=False)

    assert resolved == local_cfg


def test_default_split_config_prefers_server_on_posix(tmp_path: Path) -> None:
    splits_dir = tmp_path / "configs" / "splits"
    splits_dir.mkdir(parents=True)
    local_cfg = splits_dir / "xinyang_7_2_1.yaml"
    server_cfg = splits_dir / "xinyang_7_2_1_server.yaml"
    local_cfg.write_text("local\n", encoding="utf-8")
    server_cfg.write_text("server\n", encoding="utf-8")

    resolved = default_split_config_path(tmp_path, prefer_server=True)

    assert resolved == server_cfg
