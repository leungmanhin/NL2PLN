"""
Shared logging setup for the optimization scripts.

Mirrors stdout/stderr to a log file in addition to the original streams,
and configures the Python root logger to write to both console and the
log file.  Useful when running on Colab (or any web notebook) where the
streaming cell output truncates after a few thousand lines and progress
is otherwise lost.

Usage:
    from logging_utils import setup_logging
    setup_logging(args.log_file, args.log_level)

This captures three kinds of output to the log file:
  1. Raw print() statements (via the stdout tee)
  2. Structured logging from logging.getLogger(...).info(...) etc.
  3. Anything written to stderr (warnings, tracebacks, tqdm bars)

The pattern is borrowed from bootstrap_chainer.py, which has used it
since the start of the project; this module just makes the same
behavior available to the four optimization scripts without copy-paste.
"""
import logging
import sys


class _Tee:
    """
    Mirror writes to two file-likes (typically a stream + a log file).
    """
    def __init__(self, original, log_fh):
        self._original = original
        self._log_fh = log_fh
    def write(self, data):
        self._original.write(data)
        self._log_fh.write(data)
        self._log_fh.flush()
    def flush(self):
        self._original.flush()
        self._log_fh.flush()
    def fileno(self):
        return self._original.fileno()
    def isatty(self):
        return self._original.isatty()


def setup_logging(log_file: str, log_level: str = "info") -> None:
    """
    Configure root logging + tee stdout/stderr to ``log_file``.

    Args:
        log_file: Path to the log file.  Truncated on creation, then both
            the FileHandler and the stdout/stderr tee append to it for
            the rest of the run.
        log_level: One of "debug", "info", "warning", "error".  Sets the
            root logger level AND the console handler level.  The file
            handler is always at DEBUG so verbose details survive even
            when the console is filtered.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console_h = logging.StreamHandler()
    console_h.setLevel(level)
    console_h.setFormatter(fmt)
    root.addHandler(console_h)

    file_h = logging.FileHandler(log_file, mode="w")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    # Suppress the noisiest lower-level loggers unless DEBUG was requested.
    # Without this, httpx/LiteLLM emit one INFO line per HTTP retry and
    # quickly drown the actual optimizer progress.
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    _log_fobj = open(log_file, "a")
    sys.stdout = _Tee(sys.__stdout__, _log_fobj)
    sys.stderr = _Tee(sys.__stderr__, _log_fobj)
