import json
import os
from typing import Optional

# Stored alongside the backend package
_USERS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "users.json")
_USERS_FILE = os.path.abspath(_USERS_FILE)


def _load() -> dict:
    if not os.path.exists(_USERS_FILE):
        return {}
    with open(_USERS_FILE) as f:
        return json.load(f)


def _save(users: dict) -> None:
    with open(_USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def get_user(username: str) -> Optional[dict]:
    return _load().get(username)


def user_exists(username: str) -> bool:
    return username in _load()


def create_user(username: str, hashed_password: str) -> dict:
    users = _load()
    record = {"username": username, "hashed_password": hashed_password}
    users[username] = record
    _save(users)
    return record
