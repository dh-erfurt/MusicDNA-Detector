"""Command-line entry point for MusicDNA Detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from .analysis import analyze_file
from .audio import AudioDecodingError
from .models import AnalysisConfig, AnalysisProfileName


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract MusicDNA note events from a monophonic audio file"
    )
    parser.add_argument("input", type=Path, help="audio file to analyze")
    parser.add_argument(
        "--profile",
        choices=("default", "singing", "humming"),
        default="default",
        help="analysis profile (default: default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this file instead of stdout",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation width (default: 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Analyze one audio file and emit a ``musicdna-analysis-v0`` JSON result."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        profile = cast(AnalysisProfileName, args.profile)
        result = analyze_file(args.input, AnalysisConfig(profile=profile))
        output = result.to_json(indent=args.indent)
        if args.output is None:
            print(output)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
    except (AudioDecodingError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
