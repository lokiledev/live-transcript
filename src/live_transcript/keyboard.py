# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Single-key controls while a capture is running."""

import select
import sys
import termios
import tty

_TermiosAttr = list[int | list[bytes | int]] | list[int | list[bytes]] | list[int | list[int]]


class Keyboard:
    """Read q, p, and h without waiting for Enter."""

    def __init__(self) -> None:
        """Prepare a controller that is not yet attached to the terminal."""
        self._previous: _TermiosAttr | None = None
        self.active = False

    def start(self) -> bool:
        """Switch the terminal to cbreak mode. Return False when stdin is not a tty."""
        if not sys.stdin.isatty():
            return False
        try:
            self._previous = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except termios.error as exc:
            print(f"Keyboard shortcuts unavailable: {exc}")
            self._previous = None
            self.active = False
            return False
        self.active = True
        return True

    def stop(self) -> None:
        """Restore the terminal."""
        if self._previous is None:
            return
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._previous)
        except termios.error:
            return
        self._previous = None
        self.active = False

    def poll(self, timeout: float = 0.01) -> str | None:
        """Return one printable character, or None."""
        try:
            readable, _write, _error = select.select([sys.stdin], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not readable:
            return None
        try:
            char = sys.stdin.read(1)
        except OSError:
            return None
        if len(char) == 1 and char.isprintable():
            return char.lower()
        return None
