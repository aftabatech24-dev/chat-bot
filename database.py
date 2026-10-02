"""SQLite persistence for chatbot accounts, sessions, notes, and chat history."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Iterator, Optional


PASSWORD_ITERATIONS = 310_000
SESSION_LIFETIME_DAYS = 30
NOTE_KINDS = frozenset({"note", "preference"})


@dataclass(frozen=True)
class User:
    id: int
    username: str


@dataclass(frozen=True)
class Session:
    user: User
    token: str


@dataclass(frozen=True)
class Note:
    id: int
    title: str
    kind: str
    content: str
    created_at: str
    updated_at: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


class ChatDatabase:
    """Owns all database access; each operation uses and closes its own connection."""

    def __init__(self, database_path: os.PathLike[str] | str = "chatbot.db") -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the schema if this is the first run."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT
                );

                CREATE INDEX IF NOT EXISTS sessions_user_id_idx
                    ON sessions(user_id);

                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('note', 'preference')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS notes_user_id_idx
                    ON notes(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS interaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    user_message TEXT NOT NULL,
                    bot_message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS history_user_id_idx
                    ON interaction_history(user_id, id DESC);
                """
            )

    @staticmethod
    def _validate_username(username: str) -> str:
        value = (username or "").strip()
        if not 3 <= len(value) <= 32:
            raise ValueError("Username must be between 3 and 32 characters.")
        if any(character.isspace() for character in value):
            raise ValueError("Username cannot contain spaces.")
        return value

    @staticmethod
    def _password_digest(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
        )

    def register_user(self, username: str, password: str) -> User:
        username = self._validate_username(username)
        if len(password or "") < 8:
            raise ValueError("Password must contain at least 8 characters.")

        salt = secrets.token_bytes(16)
        digest = self._password_digest(password, salt)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """INSERT INTO users
                       (username, password_hash, password_salt, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (username, digest.hex(), salt.hex(), _timestamp(_utc_now())),
                )
                return User(id=int(cursor.lastrowid), username=username)
        except sqlite3.IntegrityError as exc:
            raise ValueError("That username is already registered.") from exc

    def login_user(self, username: str, password: str) -> Optional[Session]:
        """Verify credentials and create a session with an opaque bearer token."""
        username = (username or "").strip()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT id, username, password_hash, password_salt
                   FROM users WHERE username = ?""",
                (username,),
            ).fetchone()
            if row is None:
                return None

            try:
                salt = bytes.fromhex(row["password_salt"])
                expected = bytes.fromhex(row["password_hash"])
            except (ValueError, TypeError):
                return None
            actual = self._password_digest(password or "", salt)
            if not hmac.compare_digest(actual, expected):
                return None

            token = secrets.token_urlsafe(32)
            now = _utc_now()
            connection.execute(
                """INSERT INTO sessions (token_hash, user_id, created_at, expires_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    int(row["id"]),
                    _timestamp(now),
                    _timestamp(now + timedelta(days=SESSION_LIFETIME_DAYS)),
                ),
            )
            return Session(
                user=User(id=int(row["id"]), username=str(row["username"])),
                token=token,
            )

    def get_session_user(self, token: str) -> Optional[User]:
        """Return the session owner if a token is valid, active, and unexpired."""
        if not token:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT users.id, users.username, sessions.expires_at
                   FROM sessions JOIN users ON users.id = sessions.user_id
                   WHERE sessions.token_hash = ? AND sessions.revoked_at IS NULL""",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            try:
                expires_at = datetime.fromisoformat(row["expires_at"])
            except ValueError:
                return None
            if expires_at <= _utc_now():
                connection.execute(
                    "UPDATE sessions SET revoked_at = ? WHERE token_hash = ?",
                    (_timestamp(_utc_now()), token_hash),
                )
                return None
            return User(id=int(row["id"]), username=str(row["username"]))

    def logout(self, token: str) -> None:
        if not token:
            return
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute(
                """UPDATE sessions SET revoked_at = ?
                   WHERE token_hash = ? AND revoked_at IS NULL""",
                (_timestamp(_utc_now()), token_hash),
            )

    def save_note(
        self,
        user_id: int,
        content: str,
        title: str = "Note",
        kind: str = "note",
    ) -> int:
        if kind not in NOTE_KINDS:
            raise ValueError("Note kind must be 'note' or 'preference'.")
        title = (title or "").strip()[:120] or ("Preference" if kind == "preference" else "Note")
        content = (content or "").strip()
        if not content:
            raise ValueError("Note content cannot be empty.")
        now = _timestamp(_utc_now())
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO notes (user_id, title, kind, content, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (int(user_id), title, kind, content, now, now),
            )
            return int(cursor.lastrowid)

    def update_note(self, user_id: int, note_id: int, title: str, content: str, kind: str) -> bool:
        if kind not in NOTE_KINDS:
            raise ValueError("Note kind must be 'note' or 'preference'.")
        title = (title or "").strip()[:120] or ("Preference" if kind == "preference" else "Note")
        content = (content or "").strip()
        if not content:
            raise ValueError("Note content cannot be empty.")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE notes SET title = ?, kind = ?, content = ?, updated_at = ?
                   WHERE id = ? AND user_id = ?""",
                (title, kind, content, _timestamp(_utc_now()), int(note_id), int(user_id)),
            )
            return cursor.rowcount > 0

    def delete_note(self, user_id: int, note_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM notes WHERE id = ? AND user_id = ?",
                (int(note_id), int(user_id)),
            )
            return cursor.rowcount > 0

    def list_notes(self, user_id: int, kind: Optional[str] = None) -> list[Note]:
        if kind is not None and kind not in NOTE_KINDS:
            raise ValueError("Note kind must be 'note' or 'preference'.")
        with self._connect() as connection:
            if kind is None:
                rows = connection.execute(
                    """SELECT id, title, kind, content, created_at, updated_at FROM notes
                       WHERE user_id = ? ORDER BY updated_at DESC, id DESC""",
                    (int(user_id),),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT id, title, kind, content, created_at, updated_at FROM notes
                       WHERE user_id = ? AND kind = ? ORDER BY updated_at DESC, id DESC""",
                    (int(user_id), kind),
                ).fetchall()
        return [Note(**dict(row)) for row in rows]

    def save_preference(self, user_id: int, key: str, value: str) -> int:
        title = (key or "").strip()[:120] or "Preference"
        content = (value or "").strip()
        if not content:
            raise ValueError("Preference value cannot be empty.")
        now = _timestamp(_utc_now())
        with self._connect() as connection:
            existing = connection.execute(
                """SELECT id FROM notes
                   WHERE user_id = ? AND kind = 'preference' AND title = ? COLLATE NOCASE
                   ORDER BY id DESC LIMIT 1""",
                (int(user_id), title),
            ).fetchone()
            if existing is not None:
                note_id = int(existing["id"])
                connection.execute(
                    """UPDATE notes SET title = ?, content = ?, updated_at = ? WHERE id = ?""",
                    (title, content, now, note_id),
                )
                return note_id
            cursor = connection.execute(
                """INSERT INTO notes (user_id, title, kind, content, created_at, updated_at)
                   VALUES (?, ?, 'preference', ?, ?, ?)""",
                (int(user_id), title, content, now, now),
            )
            return int(cursor.lastrowid)

    def get_preferences(self, user_id: int) -> dict[str, str]:
        preferences: dict[str, str] = {}
        for note in self.list_notes(user_id, "preference"):
            preferences.setdefault(note.title, note.content)
        return preferences

    def save_interaction(self, user_id: int, user_message: str, bot_message: str) -> int:
        now = _timestamp(_utc_now())
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO interaction_history
                   (user_id, user_message, bot_message, created_at)
                   VALUES (?, ?, ?, ?)""",
                (int(user_id), user_message, bot_message, now),
            )
            return int(cursor.lastrowid)

    def list_interactions(self, user_id: int, limit: int = 50) -> list[dict[str, str]]:
        safe_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT user_message, bot_message, created_at FROM interaction_history
                   WHERE user_id = ? ORDER BY id DESC LIMIT ?""",
                (int(user_id), safe_limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]
