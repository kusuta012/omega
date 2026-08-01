from __future__ import annotations
import argparse
import asyncio
from code import interact
from collections.abc import Sequence
from omega import __version__
from omega.cli.doctor import print_doctor_report, run_doctor
from omega.cli.log_viewer import show_logs
from omega.cli.logging_config import configure_logging
from omega.cli.repl import run_repl
from omega.cli.tui import run_tui

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omega",
        description="Omega, a local-first personal memory agent.",
    )
    parser.add_argument("--version", action="version", version=f"Omega {__version__}")
    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the latest session (available with the interactive client)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use the scrolling REPL instead of the full-screen TUI",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="run independent environment health checks.")

    logs_parser = subparsers.add_parser("logs", help="print the persisted TUI log file")
    logs_parser.add_argument("--follow", action="store_true", help="keep printing appended log lines")
    logs_parser.add_argument("--lines", type=int, default=100, help="Number of recent lines to show.")
    return parser

def _interactive_not_ready(parser: argparse.ArgumentParser) -> int:
    parser.error(
        "The full-screen TUI is not available yet, Use 'omega --cli for the streaming"
        "terminal interface or 'omega doctor' / 'omega logs' for currently available commands."
    )
    return 2

def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    interactive_tui = args.command is None and not args.cli
    configure_logging(mode="tui" if interactive_tui else "cli")

    if args.command == "doctor":
        return 0 if print_doctor_report(asyncio.run(run_doctor())) else 1
    if args.command == "logs":
        try:
            return show_logs(line_count=args.lines, follow=args.follow)
        except ValueError as exc:
            parser.error(str(exc))
            return 2
    if args.cli:
        return asyncio.run(run_repl(continue_session=args.continue_session))
    return asyncio.run(run_tui(continue_session=args.continue_session))


if __name__ == "__main__":
    raise SystemExit(main())