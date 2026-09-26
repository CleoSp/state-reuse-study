"""Headless entry point. The driver owns sequencing; callers only read its log."""
from state_repair.execution.driver import main

if __name__ == "__main__":
    raise SystemExit(main())
