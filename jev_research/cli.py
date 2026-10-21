"""Command-line entry points; inference is explicit and never starts on import."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/jev_study"))
    parser.add_argument("--results", type=Path, default=Path("research/results/jev_error_study"))
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Fit/reuse full-data XGBoost and export the 100-error study")
    prepare.add_argument("--csv", type=Path, default=os.environ.get("SUICIDE_DATA_PATH"))
    commands.add_parser("evaluate-base", help="Spend up to 40 paid attempts on prespecified evaluation errors")
    commands.add_parser("report", help="Rebuild redacted reports offline; no model calls")
    args = parser.parse_args(argv)
    from .study import prepare_study, report_saved_results, run_base_evaluation
    if args.command == "prepare":
        if args.csv is None:
            parser.error("prepare requires --csv or SUICIDE_DATA_PATH")
        value = prepare_study(args.csv, args.artifacts, args.results)
    elif args.command == "evaluate-base":
        value = run_base_evaluation(args.artifacts, args.results)
    else:
        value = report_saved_results(args.artifacts, args.results)
    # CLI summaries contain only redacted metadata/aggregates.
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
