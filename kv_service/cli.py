"""Command-line entry point for deterministic generation reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from .checkpoint import store_lock
from .service import GenerationService

CRASH_EXIT_CODE = 75


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen TinyLM request schedule")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--journal")
    parser.add_argument("--crash-after-batches", type=int)
    parser.add_argument("--store")
    parser.add_argument("--hold-ms", type=int)
    parser.add_argument("--no-cache", action="store_true")
    return parser


def _usage_error(message: str) -> int:
    print(f"generation input error: {message}", file=sys.stderr)
    return 2


def _remove(path: Path) -> None:
    path.unlink(missing_ok=True)


def _write_report(output_path: Path, report: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output_path)


def _run_with_store(args: argparse.Namespace, document: object) -> dict:
    store_path = Path(args.store)
    with store_lock(store_path):
        service = GenerationService()
        if store_path.exists():
            service.load_checkpoint(store_path)
        run = service.begin_run(document)
        if args.hold_ms:
            time.sleep(args.hold_ms / 1000)
        executed = 0
        while not run.done:
            service.run_next_batch(run)
            executed += 1
            if args.crash_after_batches is not None and executed >= args.crash_after_batches:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(CRASH_EXIT_CODE)
        report = service.report(run)
        service.save_checkpoint(store_path)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    journal_path = Path(args.journal) if args.journal else None
    if args.store is not None:
        if journal_path is not None or args.no_cache or args.checkpoint_in or args.checkpoint_out:
            return _usage_error("--store cannot be combined with --journal, --no-cache, or checkpoint options")
    elif args.hold_ms is not None:
        return _usage_error("--hold-ms needs --store")
    if args.hold_ms is not None and args.hold_ms < 0:
        return _usage_error("--hold-ms must not be negative")
    if args.crash_after_batches is not None and (
            (journal_path is None and args.store is None) or args.crash_after_batches < 1):
        return _usage_error("--crash-after-batches needs --journal or --store and a positive count")
    if journal_path is not None and (args.no_cache or args.checkpoint_in):
        return _usage_error("--journal cannot be combined with --no-cache or --checkpoint-in")
    try:
        with Path(args.input).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if args.store is not None:
            report = _run_with_store(args, document)
            _write_report(output_path, report)
            return 0
        service = GenerationService()
        if args.checkpoint_in:
            service.load_checkpoint(args.checkpoint_in)
        if journal_path is not None and journal_path.exists():
            run = service.resume_run(document, journal_path)
        else:
            run = service.begin_run(document, cache_enabled=not args.no_cache)
        executed = 0
        while not run.done:
            service.run_next_batch(run)
            executed += 1
            if journal_path is not None:
                service.save_journal(journal_path, document, run)
                if args.crash_after_batches is not None and executed >= args.crash_after_batches:
                    return CRASH_EXIT_CODE
        report = service.report(run)
        if args.checkpoint_out:
            service.save_checkpoint(args.checkpoint_out)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"generation input error: {exc}", file=sys.stderr)
        return 2
    _write_report(output_path, report)
    if journal_path is not None:
        _remove(journal_path)
        _remove(journal_path.with_name(journal_path.name + ".tmp"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
