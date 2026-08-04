from __future__ import annotations
import argparse
import asyncio
import sys
from collections.abc import Sequence
from omega import __version__   
from omega.cli.doctor import print_doctor_report, run_doctor
from omega.cli.lifecycle import run_new_session, run_uninstall, run_ingest, run_kb_command
from omega.cli.log_viewer import show_logs
from omega.cli.logging_config import configure_logging
from omega.cli.repl import run_repl
from omega.cli.tui import run_tui
from omega.cli.config_wizard import run_config
from omega.cli.setup_wizard import run_setup
from omega.storage.postgres_session import db_pool

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
    subparsers.add_parser("setup", help="Setup Omega and validate local dependencies.")
    subparsers.add_parser("config", help="Update and validate LLM provider settings")
    subparsers.add_parser("doctor", help="run independent environment health checks.")
    subparsers.add_parser("new", help="start a new persisted session")
    ingest_parser = subparsers.add_parser("ingest", help="queue a URL, text, or code item for the knowledge base")
    ingest_parser.add_argument("source_type", choices=["url", "text", "code"])
    ingest_parser.add_argument("content", nargs="?", help="URL, text, or code content")
    ingest_parser.add_argument("--file", dest="file_path", help="Read text or code content from a UTF-8 file")
    ingest_parser.add_argument("--title", help="Optional item title")
    kb_parser = subparsers.add_parser("kb", help="inspect and manage knowledge-base items")
    kb_subparsers = kb_parser.add_subparsers(dest="kb_command", required=True)
    kb_list_parser = kb_subparsers.add_parser("list", help="list knowledge-base items")
    kb_list_parser.add_argument("--limit", type=int, default=20, choices=range(1, 101))
    kb_list_parser.add_argument("--status", choices=["pending", "running", "done", "failed"])
    uninstall_parser = subparsers.add_parser("uninstall", help="remove the installed Omega package")
    uninstall_parser.add_argument("--yes", action="store_true", help="confirm the package removal")

    logs_parser = subparsers.add_parser("logs", help="print the persisted TUI log file")
    logs_parser.add_argument("--follow", action="store_true", help="keep printing appended log lines")
    logs_parser.add_argument("--lines", type=int, default=100, help="Number of recent lines to show.")
    return parser

async def _run_with_database(run_client) -> int:
    await db_pool.connect()
    try:
        return await run_client()
    finally:
        await db_pool.disconnect()

def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is not None and (args.cli or args.continue_session):
        parser.error("--cli and --continue are available only for the interactive client")
    interactive_tui = args.command is None and not args.cli
    configure_logging(mode="tui" if interactive_tui else "cli")

    if args.command == "setup":
        return run_setup()
    if args.command == "config":
        return run_config()
    if args.command == "uninstall":
        return run_uninstall(yes=args.yes)
    if args.command == "new":
        return run_new_session()
    if args.command == "ingest":
        return run_ingest(args.source_type, args.content, args.title, args.file_path)
    if args.command == "kb":
        return run_kb_command(
            args.kb_command,
            getattr(args, "item_id", None),
            getattr(args, "limit", 20),
            getattr(args, "status", None),
            getattr(args, "yes", False),
        )
    if args.command == "doctor":
        return 0 if print_doctor_report(asyncio.run(run_doctor())) else 1
    if args.command == "logs":
        try:
            return show_logs(line_count=args.lines, follow=args.follow)
        except ValueError as exc:
            parser.error(str(exc))
            return 2
    if args.cli:
        run_client = lambda: run_repl(continue_session=args.continue_session)
    else:
        run_client = lambda: run_tui(continue_session=args.continue_session)
    try:
        return asyncio.run(_run_with_database(run_client))
    except KeyboardInterrupt:
        print("\nOmega stopped", file=sys.stderr)
        return 130
    except Exception as err:
        print(f"Omega could not start: {err}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())