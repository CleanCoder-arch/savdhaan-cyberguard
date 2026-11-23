"""Run a repeatable, local benchmark on grouped URL and SMS data."""

import argparse
import csv
from pathlib import Path

from . import DEFAULT_RESULTS_PATH
from .evaluate import read_csv, run_benchmark, write_results
from .split_builder import LabeledExample


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Input files are UTF-8 CSV with text,label headers. Labels are 0 for benign data and 1 for scams.",
    )
    parser.add_argument("--url-data", type=Path, help="URL CSV with absolute HTTP(S) URLs")
    parser.add_argument("--sms-data", type=Path, help="SMS CSV with message text")
    parser.add_argument("--demo", action="store_true", help="Run on the bundled synthetic dataset")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH, help="Destination JSON path")
    args = parser.parse_args()
    if args.demo and (args.url_data is not None or args.sms_data is not None):
        parser.error("--demo cannot be combined with input files.")
    if not args.demo and (args.url_data is None or args.sms_data is None):
        parser.error("Specify --demo or provide both --url-data and --sms-data.")
    try:
        if args.demo:
            from scripts.bootstrap_models import _SMS_EXAMPLES, _URL_EXAMPLES

            urls = tuple(LabeledExample(text, label) for label, text in _URL_EXAMPLES)
            messages = tuple(LabeledExample(text, label) for label, text in _SMS_EXAMPLES)
        else:
            if any(path.resolve() == args.output.resolve() for path in (args.url_data, args.sms_data)):
                parser.error("The output path must differ from the input paths.")
            urls = read_csv(args.url_data)
            messages = read_csv(args.sms_data)
        result = run_benchmark(urls, messages, seed=args.seed, dataset_kind="synthetic_demo" if args.demo else "provided")
        write_results(args.output, result)
    except (UnicodeError, csv.Error):
        parser.error("The benchmark input is not readable UTF-8 CSV.")
    except OSError:
        parser.error("The benchmark input or output file could not be accessed.")
    except ValueError as error:
        parser.error(str(error))
    print(f"Benchmark results written to {args.output}")


if __name__ == "__main__":
    main()
