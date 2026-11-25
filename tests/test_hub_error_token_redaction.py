# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A token pasted twice with a CRLF between ("hf_FAKE\\r\\nhf_FAKE") made h11 refuse the Authorization header, and the
exception text carried the whole token. check_hf_model_exists then copied that text into its RuntimeError and chained
the original, so the token printed three times. A malformed token must be refused up front, and no token may reach a
user facing message or traceback. Only fake tokens appear here."""

import traceback

import h11
import pytest
import requests

from unsloth_zoo import saving_utils
from unsloth_zoo.hf_utils import _clean_hf_token, _scrub_secrets

FAKE = "hf_FAKE"
DOUBLE_PASTE = f"{FAKE}\r\n{FAKE}"


def _h11_illegal_header_error():
    """The error h11 really raises for the double paste (at Request construction), produced offline."""
    try:
        h11.Request(
            method = "GET", target = "/",
            headers = [("Host", "huggingface.co"), ("Authorization", f"Bearer {DOUBLE_PASTE}")],
        )
    except h11.LocalProtocolError as error:
        return error
    raise AssertionError("h11 accepted a CRLF in a header value")


def _full_text(error):
    """Everything Python prints for `error`: message, chained causes and contexts."""
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def _patch_ls(monkeypatch, side_effect):
    calls = []

    def fake_ls(self, path, detail = True, **kwargs):
        calls.append(path)
        raise side_effect
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    return calls


def test_the_fixture_matches_the_reported_error():
    error = _h11_illegal_header_error()
    assert "Illegal header value" in str(error)
    assert FAKE in str(error)


# Validation before any request.

def test_double_pasted_token_is_refused_before_any_request(monkeypatch):
    calls = _patch_ls(monkeypatch, AssertionError("the Hub must not be asked"))

    with pytest.raises(ValueError) as excinfo:
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct", token = DOUBLE_PASTE)

    text = _full_text(excinfo.value)
    assert "malformed" in text
    assert "whitespace" in text
    assert FAKE not in text
    assert calls == []


@pytest.mark.parametrize("token", [f"{FAKE} {FAKE}", "hf_FA\tKE", "hf_FA\x00KE", "hf_FA\u200bKE"])
def test_other_embedded_whitespace_or_control_characters_are_refused(monkeypatch, token):
    _patch_ls(monkeypatch, AssertionError("the Hub must not be asked"))
    with pytest.raises(ValueError, match = "malformed"):
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct", token = token)


def test_local_path_does_not_validate_the_token(monkeypatch):
    # A local directory never uses the token, so a bad one must not turn a False into a crash.
    assert saving_utils.check_hf_model_exists("./not-a-hub-repo", token = DOUBLE_PASTE) is False


@pytest.mark.parametrize("raw", [f"{FAKE}\n", f"{FAKE}\r\n", f"  {FAKE}  "])
def test_surrounding_whitespace_is_stripped(monkeypatch, raw):
    seen = {}

    class FakeFileSystem:
        def __init__(self, token = None, **kwargs):
            seen["token"] = token

        def ls(self, path, detail = True, **kwargs):
            return [{"name": "unsloth/x/model.safetensors", "size": 1}]
    monkeypatch.setattr(saving_utils, "HfFileSystem", FakeFileSystem)

    assert saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct", token = raw) is True
    assert seen["token"] == FAKE


@pytest.mark.parametrize("token", [None, True, False])
def test_non_string_tokens_pass_through(token):
    assert _clean_hf_token(token) is token


def test_clean_token_does_not_echo_the_value():
    with pytest.raises(ValueError) as excinfo:
        _clean_hf_token(DOUBLE_PASTE)
    assert FAKE not in str(excinfo.value)


# A bad header from some other source (HF_TOKEN, a cached token) still must not leak.

def test_h11_header_error_message_and_traceback_carry_no_token(monkeypatch):
    _patch_ls(monkeypatch, _h11_illegal_header_error())

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")

    error = excinfo.value
    assert FAKE not in _full_text(error)
    assert FAKE not in str(error)
    assert error.__cause__ is None
    assert "malformed" in str(error)
    assert "Illegal header value" in str(error)


def test_header_error_is_not_called_a_connectivity_problem(monkeypatch):
    _patch_ls(monkeypatch, _h11_illegal_header_error())

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")

    assert "not a connectivity problem" in str(excinfo.value)
    assert "Retry" not in str(excinfo.value)


def test_requests_invalid_header_error_carries_no_token(monkeypatch):
    error = requests.exceptions.InvalidHeader(
        f"Invalid leading whitespace, reserved character(s), or return character(s) in header value: "
        f"'Bearer {DOUBLE_PASTE}'"
    )
    _patch_ls(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")

    assert FAKE not in _full_text(excinfo.value)
    assert "malformed" in str(excinfo.value)


def test_secret_in_a_wrapped_cause_still_breaks_the_chain(monkeypatch):
    # The wrapper's own text is clean; the secret is only in what it was raised from.
    try:
        try:
            raise _h11_illegal_header_error()
        except h11.LocalProtocolError as inner:
            raise OSError("request failed") from inner
    except OSError as wrapped:
        error = wrapped
    _patch_ls(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")

    assert FAKE not in _full_text(excinfo.value)


def test_a_real_connectivity_error_keeps_its_message_and_its_chain(monkeypatch):
    original = ConnectionError("Temporary failure in name resolution")
    _patch_ls(monkeypatch, original)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")

    assert "connectivity or rate limiting problem" in str(excinfo.value)
    assert "Temporary failure in name resolution" in str(excinfo.value)
    assert excinfo.value.__cause__ is original


def test_other_hub_unreachable_call_sites_scrub_too():
    error = saving_utils._hub_unreachable_error(
        "unsloth/x", ConnectionError(f"401 for Bearer {DOUBLE_PASTE}"),
        action = "reading the quantization config of `unsloth/x`", mistaken_for = "an unquantized model",
    )
    assert FAKE not in str(error)
    assert "Bearer ***" in str(error)
    assert "an unquantized model" in str(error)


def test_gated_repo_warning_carries_no_token():
    with pytest.warns(UserWarning) as record:
        saving_utils._warn_gated_repo("org/gated", RuntimeError(f"403 Bearer {FAKE}"))
    assert FAKE not in str(record[0].message)


# The scrubber itself.

@pytest.mark.parametrize("text", [
    FAKE,
    f"token {FAKE}.",
    f"({FAKE})",
    f"b'Bearer {FAKE}'",
    f"Illegal header value b'Bearer {DOUBLE_PASTE}'",
    f"Illegal header value {_h11_illegal_header_error().args[0]!r}",
    f"Authorization: Bearer {DOUBLE_PASTE}\r\nHost: huggingface.co",
    "Authorization: bearer abc123\r\ndef456",
    "BEARER abc.def-ghi_jkl",
])
def test_scrub_removes_every_credential(text):
    scrubbed = _scrub_secrets(text)
    for secret in (FAKE, "abc123", "def456", "abc.def-ghi_jkl"):
        assert secret not in scrubbed
    assert "***" in scrubbed


def test_scrub_keeps_the_surrounding_diagnosis():
    text = f"Illegal header value b'Bearer {DOUBLE_PASTE}'"
    assert _scrub_secrets(text).startswith("Illegal header value b'")


@pytest.mark.parametrize("text", [
    "hf_hub_download failed",
    "see hf_xet_fallback for details",
    "Connection reset by peer",
    "carrier bearer-bond",
])
def test_scrub_leaves_ordinary_text_alone(text):
    assert _scrub_secrets(text) == text
