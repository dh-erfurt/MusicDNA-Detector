from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).parents[1]


def _pyproject() -> dict[str, object]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _public_file(name: str) -> Path:
    public_name = f"{Path(name).stem}.public{Path(name).suffix}"
    public_path = REPOSITORY_ROOT / public_name
    return public_path if public_path.exists() else REPOSITORY_ROOT / name


def test_experimental_release_identity_is_consistent() -> None:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    assert project["version"] == "0.9.0"
    assert project["license"] == "MIT AND ISC"
    assert project["authors"] == [
        {"name": "Anna Neovesky"},
        {"name": "Finn Johann Romeis"},
    ]
    scripts = project["scripts"]
    assert isinstance(scripts, dict)
    assert scripts["musicdna-detector"] == "musicdna_detector.cli:main"
    assert "Experimental" in project["description"]
    assert project["readme"] == "README.rst"
    assert "Development Status :: 3 - Alpha" in project["classifiers"]

    citation = (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert 'version: "0.9.0"' in citation
    assert 'version: "0.5.1"' not in citation
    assert "date-released: 2026-09-15" in citation
    assert "family-names: Neovesky" in citation
    assert "family-names: Romeis" in citation
    assert "## [0.9.0] — unreleased" in changelog
    assert "## [0.5.1]" not in changelog

    public_changelog = _public_file("CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [0.9.0] — unreleased" in public_changelog
    assert "Initial public release." in public_changelog


def test_public_readme_covers_the_package_entry_contract() -> None:
    readme = _public_file("README.rst").read_text(encoding="utf-8")
    for required in (
        "This is release ``0.9.0``",
        "pre-1.0",
        "python -m pip install .",
        "musicdna-detector @ git+https://github.com/dh-erfurt/MusicDNA-Detector.git",
        "musicdna-detector path/to/query.wav",
        "--output detector-output.json",
        "AnalysisConfig",
        "analyze_file",
        "examples/synthetic_melody.wav",
        "no required audio directory",
        "repository root",
        "``musicdna-analysis-v0``",
        "MusicDNA-Encoder",
        "musicdna-detector[librosa]",
        "HumTrans",
        "VocalSet audio release",
        "Annotated-VocalSet",
        "Known limits",
        "THIRD_PARTY_NOTICES.txt",
    ):
        assert required in readme


def test_librosa_reference_backend_is_an_official_extra() -> None:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    assert set(optional) == {"dev", "librosa"}
    assert optional["librosa"] == ["librosa>=0.11,<0.12"]


def test_third_party_notice_relies_on_license_file_metadata_only() -> None:
    pyproject = _pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)
    assert project["license-files"].count("THIRD_PARTY_NOTICES.txt") == 1

    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "force-include" not in wheel
