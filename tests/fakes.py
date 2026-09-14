"""Fake credentials for scanner tests, assembled at runtime.

The source code never contains a contiguous secret-shaped string, so the repository secret
scan stays clean while the tests still exercise every rule. None of these are real.
"""

from __future__ import annotations

import random
import string


def _join(*parts: str) -> str:
    return "".join(parts)


def fake_private_key_header() -> str:
    return _join("-----BEGIN ", "RSA ", "PRIVATE ", "KEY-----")


def fake_aws_key_id() -> str:
    return _join("AK", "IA", "QZ7X", "K2M9", "P4TN", "W8RB")


def fake_github_token() -> str:
    return _join("gh", "p_", "Aa1Bb2Cc3" * 4)


def fake_sk_key() -> str:
    return _join("sk", "-proj-", "Tq8Lm3Xz9Rp2" * 2)


def fake_slack_token() -> str:
    return _join("xo", "xb-", "1234567890-", "AbCdEfGhIj")


def fake_google_key() -> str:
    return _join("AI", "za", "Sy", "B1c2D3e4F5g6H7i8", "J9k0L1m2N3o4P5q6r")


def fake_discord_token() -> str:
    return _join("M", "Tk4" * 8, ".", "Gx7Yq2", ".", "Zr5Wp8Kd3Lm9" * 3)


def fake_telegram_token() -> str:
    return _join("123456789", ":", "AA", "Hq7Lz2Wx9Km4Rt6Y", "p1Nc8Vb3Jd5Fg0Ss", "e")


def fake_quoted_assignment() -> str:
    return _join("api_key", ' = "', "Zx9Qw8Er7Ty6", '"')


def fake_env_assignment() -> str:
    return _join("DISCORD_BOT_", "TOKEN", "=", "Pq7Wm2Xz8Lk4Rn6")


def fake_high_entropy_literal() -> str:
    rng = random.Random(1337)
    alphabet = string.ascii_letters + string.digits + "+/"
    value = "".join(rng.choice(alphabet) for _ in range(44))
    return _join('blob = "', value, '"')


ALL_FAKES: dict[str, str] = {
    "private_key_block": fake_private_key_header(),
    "aws_access_key_id": fake_aws_key_id(),
    "github_token": fake_github_token(),
    "sk_api_key": fake_sk_key(),
    "slack_token": fake_slack_token(),
    "google_api_key": fake_google_key(),
    "discord_bot_token": fake_discord_token(),
    "telegram_bot_token": fake_telegram_token(),
    "credential_assignment": fake_quoted_assignment(),
    "high_entropy_string": fake_high_entropy_literal(),
}
