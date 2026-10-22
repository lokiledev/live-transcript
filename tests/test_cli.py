# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Command-line defaults."""

import pytest

from live_transcript.cli import build_parser


def test_a_capture_is_one_language() -> None:
    parser = build_parser()
    english = parser.parse_args(["live"])
    french = parser.parse_args(["visio", "--language", "fr"])
    assert english.language == "en"
    assert french.language == "fr"
    with pytest.raises(SystemExit):
        parser.parse_args(["live", "--language", "zh"])


def test_reprocess_keeps_the_saved_language_and_model_by_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["reprocess", "latest"])
    assert args.language is None
    assert args.model is None
    overridden = parser.parse_args(["reprocess", "latest", "--language", "fr", "--model", "small"])
    assert overridden.language == "fr"
    assert overridden.model == "small"
