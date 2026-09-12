from __future__ import annotations

import json
from pathlib import Path

from musicdna_detector import AnalysisConfig, cli


class _FakeResult:
    def to_json(self, *, indent: int | None = None) -> str:
        assert indent == 0
        return json.dumps({"schema_version": "musicdna-analysis-v0"})


def test_cli_writes_json_to_stdout(monkeypatch, tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "query.wav"
    input_path.touch()
    received: dict[str, object] = {}

    def fake_analyze(path: Path, config: AnalysisConfig) -> _FakeResult:
        received["path"] = path
        received["profile"] = config.profile
        return _FakeResult()

    monkeypatch.setattr(cli, "analyze_file", fake_analyze)

    assert cli.main([str(input_path), "--profile", "humming", "--indent", "0"]) == 0

    assert received == {"path": input_path, "profile": "humming"}
    assert json.loads(capsys.readouterr().out)["schema_version"] == "musicdna-analysis-v0"


def test_cli_writes_json_to_output_file(monkeypatch, tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "query.wav"
    output_path = tmp_path / "nested" / "detector-output.json"
    input_path.touch()
    monkeypatch.setattr(cli, "analyze_file", lambda path, config: _FakeResult())

    assert cli.main([str(input_path), "--output", str(output_path), "--indent", "0"]) == 0

    assert output_path.read_text(encoding="utf-8") == '{"schema_version": "musicdna-analysis-v0"}\n'
    assert capsys.readouterr().out == ""
