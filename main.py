"""Tkinter chatbot application with AIML rules, ML fallback, and SQLite storage."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Iterator, Optional

from database import ChatDatabase, Note, Session, User
from ml_fallback import IntentClassifier


BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "chatbot.db"
BRAIN_FILE = BASE_DIR / "bot_brain.brn"
STARTUP_FILE = BASE_DIR / "std-startup.xml"
AIML_FALLBACK_MARKER = "__ML_FALLBACK__"

APP_BACKGROUND = "#EEF3F9"
HEADER_BACKGROUND = "#1E3A5F"
PRIMARY_BLUE = "#2563EB"
PRIMARY_HOVER = "#1D4ED8"
CHAT_WHITE = "#FFFFFF"
CHAT_BACKGROUND = "#F8FAFC"
INPUT_GRAY = "#F1F5F9"
USER_BUBBLE = "#DBEAFE"
BOT_BUBBLE = "#DCFCE7"
USER_TEXT = "#1E40AF"
BOT_TEXT = "#166534"
TEXT_COLOR = "#1F2937"
FONT = ("Arial", 12)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@contextmanager
def _project_working_directory() -> Iterator[None]:
    """Resolve relative AIML <learn> paths beside this script during boot only."""
    old_directory = Path.cwd()
    os.chdir(BASE_DIR)
    try:
        yield
    finally:
        os.chdir(old_directory)


def _load_compiled_brain(kernel: object, brain_file: Path) -> None:
    bootstrap = getattr(kernel, "bootstrap", None)
    if callable(bootstrap):
        bootstrap(brainFile=str(brain_file))
        return
    load_brain = getattr(kernel, "loadBrain", None)
    if callable(load_brain):
        load_brain(str(brain_file))
        return
    raise RuntimeError("The installed AIML package cannot load compiled brains.")


def _learn_startup_files(kernel: object, startup_file: Path) -> None:
    bootstrap = getattr(kernel, "bootstrap", None)
    startup_name = startup_file.name
    if callable(bootstrap):
        try:
            bootstrap(learnFiles=startup_name, commands="LOAD AIML B")
            return
        except TypeError:
            # Some AIML distributions expose learn/respond without bootstrap kwargs.
            pass

    learn = getattr(kernel, "learn", None)
    respond = getattr(kernel, "respond", None)
    if not callable(learn) or not callable(respond):
        raise RuntimeError("The installed AIML package cannot learn startup files.")
    learn(startup_name)
    respond("LOAD AIML B")


def initialize_aiml_kernel(
    brain_file: Path = BRAIN_FILE,
    startup_file: Path = STARTUP_FILE,
) -> Optional[object]:
    """Load a saved brain, or compile the AIML startup files on the first run."""
    try:
        import aiml  # type: ignore[import-not-found]
    except ImportError as exc:
        LOGGER.warning("AIML is unavailable (%s); using the ML fallback only.", exc)
        return None

    kernel_type = getattr(aiml, "Kernel", None)
    if kernel_type is None:
        LOGGER.warning("The installed AIML module does not expose Kernel; using ML fallback.")
        return None

    with _project_working_directory():
        if os.path.isfile(str(brain_file)):
            try:
                kernel = kernel_type()
                _load_compiled_brain(kernel, brain_file)
                LOGGER.info("Loaded compiled AIML brain from %s", brain_file.name)
                return kernel
            except Exception:
                LOGGER.exception("Could not load the saved AIML brain; rebuilding it.")

        try:
            kernel = kernel_type()
            _learn_startup_files(kernel, startup_file)
            save_brain = getattr(kernel, "saveBrain", None)
            if callable(save_brain):
                try:
                    save_brain(str(brain_file))
                    LOGGER.info("Saved compiled AIML brain to %s", brain_file.name)
                except Exception:
                    LOGGER.exception("AIML loaded, but its compiled brain could not be saved.")
            else:
                LOGGER.info("This AIML package does not support saving a compiled brain.")
            return kernel
        except Exception:
            LOGGER.exception("Could not initialize AIML; using the ML fallback only.")
            return None


class ChatService:
    """Routes a message through AIML first and ML intent handling second."""

    def __init__(self, kernel: Optional[object], classifier: Optional[IntentClassifier] = None) -> None:
        self.kernel = kernel
        self.classifier = classifier or IntentClassifier()

    def respond(self, message: str, user: User) -> str:
        clean_message = (message or "").strip()
        if not clean_message:
            return "Please enter a message."
        if len(clean_message) > 2000:
            return "Please keep messages under 2,000 characters."

        if self.kernel is not None:
            respond = getattr(self.kernel, "respond", None)
            if callable(respond):
                try:
                    result = respond(clean_message, userId=str(user.id))
                    if isinstance(result, str):
                        result = result.strip()
                        if result and result != AIML_FALLBACK_MARKER:
                            return result
                except Exception:
                    LOGGER.exception("AIML could not process a message; trying the ML fallback.")
        return self.classifier.respond(clean_message)


class ChatbotApp:
    def __init__(self, root: tk.Tk, database: ChatDatabase, service: ChatService) -> None:
        self.root = root
        self.database = database
        self.service = service
        self.session: Optional[Session] = None
        self.root.title("Chatbot")
        self.root.geometry("820x620")
        self.root.minsize(620, 460)
        self.root.configure(bg=APP_BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._show_authentication()

    def _clear_root(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()

    def _show_authentication(self) -> None:
        self.session = None
        self.root.title("Chatbot | Sign in")
        self._clear_root()
        frame = tk.Frame(self.root, bg=APP_BACKGROUND, padx=42, pady=36)
        frame.pack(expand=True, fill="both")

        tk.Label(frame, text="Welcome", bg=APP_BACKGROUND, fg=HEADER_BACKGROUND,
                 font=("Arial", 22, "bold")).pack(pady=(12, 8))
        tk.Label(frame, text="Sign in or create an account to start chatting.",
                 bg=APP_BACKGROUND, fg=TEXT_COLOR, font=FONT).pack(pady=(0, 22))

        form = tk.Frame(frame, bg=APP_BACKGROUND)
        form.pack()
        tk.Label(form, text="Username", bg=APP_BACKGROUND, fg=TEXT_COLOR, font=FONT).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        username_entry = tk.Entry(form, font=FONT, width=30, relief="solid", bd=1)
        username_entry.grid(row=1, column=0, sticky="ew", pady=(0, 14), ipady=6)
        tk.Label(form, text="Password", bg=APP_BACKGROUND, fg=TEXT_COLOR, font=FONT).grid(
            row=2, column=0, sticky="w", pady=(0, 4)
        )
        password_entry = tk.Entry(form, font=FONT, width=30, relief="solid", bd=1, show="•")
        password_entry.grid(row=3, column=0, sticky="ew", pady=(0, 18), ipady=6)

        create_account = tk.BooleanVar(value=False)
        status = tk.Label(frame, text="", bg=APP_BACKGROUND, fg="#B00020", font=("Arial", 10))
        status.pack(pady=(4, 4))

        def submit(event: Optional[tk.Event] = None) -> Optional[str]:
            username = username_entry.get().strip()
            password = password_entry.get()
            try:
                if create_account.get():
                    self.database.register_user(username, password)
                session = self.database.login_user(username, password)
                if session is None:
                    status.configure(text="Username or password is incorrect.")
                    return "break"
                self.session = session
                self._show_chat()
            except ValueError as exc:
                status.configure(text=str(exc))
            except Exception:
                LOGGER.exception("Authentication failed due to a database error.")
                status.configure(text="Could not sign in right now. Please try again.")
            return "break"

        def update_button_label() -> None:
            action_button.configure(text="Create account" if create_account.get() else "Sign in")

        tk.Checkbutton(
            frame,
            text="Create a new account",
            variable=create_account,
            command=update_button_label,
            bg=APP_BACKGROUND,
            activebackground=APP_BACKGROUND,
            font=FONT,
        ).pack(pady=(0, 8))
        action_button = tk.Button(
            frame,
            text="Sign in",
            command=submit,
            bg=PRIMARY_BLUE,
            fg=CHAT_WHITE,
            activebackground=PRIMARY_HOVER,
            activeforeground=CHAT_WHITE,
            font=FONT,
            relief="flat",
            padx=22,
            pady=8,
            cursor="hand2",
        )
        action_button.pack()
        password_entry.bind("<Return>", submit)
        username_entry.bind("<Return>", lambda _event: password_entry.focus_set())
        username_entry.focus_set()

    def _show_chat(self) -> None:
        assert self.session is not None
        self.root.title(f"Chatbot | {self.session.user.username}")
        self._clear_root()

        header = tk.Frame(self.root, bg=HEADER_BACKGROUND, padx=14, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text=f"Chatting as {self.session.user.username}",
            bg=HEADER_BACKGROUND,
            fg=CHAT_WHITE,
            font=("Arial", 12, "bold"),
        ).pack(side="left")
        tk.Button(
            header, text="Notes", command=self._open_notes, bg=PRIMARY_BLUE, fg=CHAT_WHITE,
            activebackground=PRIMARY_HOVER, activeforeground=CHAT_WHITE, font=FONT,
            relief="flat", padx=12, pady=4, cursor="hand2",
        ).pack(side="right", padx=(8, 0))
        tk.Button(
            header, text="Sign out", command=self._sign_out, bg=HEADER_BACKGROUND, fg=CHAT_WHITE,
            activebackground="#294D78", activeforeground=CHAT_WHITE,
            font=FONT, relief="solid", bd=1, padx=10, pady=4, cursor="hand2",
        ).pack(side="right")

        history_frame = tk.Frame(self.root, bg=CHAT_BACKGROUND, padx=10, pady=10)
        history_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.chat_log = tk.Text(
            history_frame,
            wrap="word",
            state="disabled",
            bg=CHAT_BACKGROUND,
            fg=TEXT_COLOR,
            font=FONT,
            relief="flat",
            padx=8,
            pady=8,
            cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.chat_log.yview)
        self.chat_log.configure(yscrollcommand=scrollbar.set)
        self.chat_log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.chat_log.tag_configure(
            "user",
            background=USER_BUBBLE,
            foreground=USER_TEXT,
            font=("Arial", 12, "bold"),
            justify="right",
            lmargin1=90,
            lmargin2=50,
            rmargin=12,
            spacing1=6,
            spacing3=6,
        )
        self.chat_log.tag_configure(
            "bot",
            background=BOT_BUBBLE,
            foreground=BOT_TEXT,
            font=("Arial", 12),
            justify="left",
            lmargin1=12,
            lmargin2=12,
            rmargin=90,
            spacing1=6,
            spacing3=6,
        )
        self._restore_history()

        input_frame = tk.Frame(self.root, bg=INPUT_GRAY, padx=12, pady=12)
        input_frame.pack(fill="x")
        self.message_entry = tk.Entry(
            input_frame, font=FONT, bg=INPUT_GRAY, relief="solid", bd=1
        )
        self.message_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.procc,
            bg=PRIMARY_BLUE,
            fg=CHAT_WHITE,
            activebackground=PRIMARY_HOVER,
            activeforeground=CHAT_WHITE,
            font=FONT,
            relief="flat",
            padx=18,
            pady=6,
            cursor="hand2",
        )
        send_button.pack(side="right")
        self.message_entry.bind("<Return>", self.procc)
        self.message_entry.focus_set()

    def _append_message(self, speaker: str, message: str, tag: str) -> None:
        self.chat_log.configure(state="normal")
        self.chat_log.insert(tk.END, f" {speaker}: {message} \n", tag)
        self.chat_log.insert(tk.END, "\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see(tk.END)

    def _restore_history(self) -> None:
        if self.session is None:
            return
        try:
            for turn in self.database.list_interactions(self.session.user.id):
                self._append_message(self.session.user.username, turn["user_message"], "user")
                self._append_message("Bot", turn["bot_message"], "bot")
        except Exception:
            LOGGER.exception("Could not restore conversation history.")

    def procc(self, event: Optional[tk.Event] = None) -> str:
        if self.session is None:
            return "break"
        message = self.message_entry.get().strip()
        if not message:
            return "break"
        self.message_entry.delete(0, tk.END)
        self._append_message(self.session.user.username, message, "user")
        response = self.service.respond(message, self.session.user)
        self._append_message("Bot", response, "bot")
        try:
            self.database.save_interaction(self.session.user.id, message, response)
        except Exception:
            LOGGER.exception("Could not save this chat turn to history.")
        self.message_entry.focus_set()
        return "break"

    def _open_notes(self) -> None:
        if self.session is None:
            return
        NotesWindow(self.root, self.database, self.session.user)

    def _sign_out(self) -> None:
        if self.session is not None:
            try:
                self.database.logout(self.session.token)
            except Exception:
                LOGGER.exception("Could not revoke the active session token.")
        self._show_authentication()

    def close(self) -> None:
        if self.session is not None:
            try:
                self.database.logout(self.session.token)
            except Exception:
                LOGGER.exception("Could not revoke the active session token on exit.")
        self.root.destroy()


class NotesWindow:
    """Small per-user editor for saved notes and preferences."""

    def __init__(self, parent: tk.Tk, database: ChatDatabase, user: User) -> None:
        self.database = database
        self.user = user
        self.window = tk.Toplevel(parent)
        self.window.title("Notes and preferences")
        self.window.geometry("650x430")
        self.window.minsize(520, 360)
        self.window.transient(parent)

        outer = tk.Frame(self.window, bg=APP_BACKGROUND, padx=12, pady=12)
        outer.pack(fill="both", expand=True)
        left = tk.Frame(outer, bg=APP_BACKGROUND)
        left.pack(side="left", fill="y", padx=(0, 12))
        tk.Label(left, text="Saved items", bg=APP_BACKGROUND, fg=HEADER_BACKGROUND, font=FONT).pack(anchor="w")
        self.items = tk.Listbox(left, width=25, font=("Arial", 10), exportselection=False)
        self.items.pack(fill="y", expand=True, pady=(6, 8))
        self.items.bind("<<ListboxSelect>>", self._select_item)
        tk.Button(left, text="New", command=self._new_item, font=FONT).pack(fill="x")
        tk.Button(left, text="Delete", command=self._delete_item, font=FONT).pack(fill="x", pady=(5, 0))

        editor = tk.Frame(outer, bg=APP_BACKGROUND)
        editor.pack(side="left", fill="both", expand=True)
        tk.Label(editor, text="Title", bg=APP_BACKGROUND, fg=TEXT_COLOR, font=FONT).pack(anchor="w")
        self.title_entry = tk.Entry(editor, font=FONT, relief="solid", bd=1)
        self.title_entry.pack(fill="x", pady=(4, 10), ipady=5)
        tk.Label(editor, text="Type", bg=APP_BACKGROUND, fg=TEXT_COLOR, font=FONT).pack(anchor="w")
        self.kind_var = tk.StringVar(value="note")
        self.kind_combo = ttk.Combobox(
            editor, textvariable=self.kind_var, values=("note", "preference"),
            state="readonly", font=FONT,
        )
        self.kind_combo.pack(fill="x", pady=(4, 10))
        tk.Label(editor, text="Content", bg=APP_BACKGROUND, fg=TEXT_COLOR, font=FONT).pack(anchor="w")
        self.content_text = tk.Text(editor, wrap="word", font=FONT, height=9, relief="solid", bd=1)
        self.content_text.pack(fill="both", expand=True, pady=(4, 10))
        tk.Button(
            editor, text="Save", command=self._save_item, bg=PRIMARY_BLUE, fg=CHAT_WHITE,
            activebackground=PRIMARY_HOVER, activeforeground=CHAT_WHITE, font=FONT,
            relief="flat", padx=14, pady=6,
        ).pack(anchor="e")

        self.notes: list[Note] = []
        self.selected_note_id: Optional[int] = None
        self._refresh()

    def _refresh(self) -> None:
        try:
            self.notes = self.database.list_notes(self.user.id)
        except Exception:
            LOGGER.exception("Could not load saved notes.")
            messagebox.showerror("Notes", "Could not load your saved notes.", parent=self.window)
            return
        self.items.delete(0, tk.END)
        for note in self.notes:
            self.items.insert(tk.END, f"{note.title}  ·  {note.kind}")

    def _select_item(self, _event: Optional[tk.Event] = None) -> None:
        selection = self.items.curselection()
        if not selection:
            return
        note = self.notes[selection[0]]
        self.selected_note_id = note.id
        self.title_entry.delete(0, tk.END)
        self.title_entry.insert(0, note.title)
        self.kind_var.set(note.kind)
        self.content_text.delete("1.0", tk.END)
        self.content_text.insert("1.0", note.content)

    def _new_item(self) -> None:
        self.selected_note_id = None
        self.items.selection_clear(0, tk.END)
        self.title_entry.delete(0, tk.END)
        self.kind_var.set("note")
        self.content_text.delete("1.0", tk.END)
        self.title_entry.insert(0, "Note")
        self.title_entry.focus_set()

    def _save_item(self) -> None:
        title = self.title_entry.get().strip()
        content = self.content_text.get("1.0", "end-1c").strip()
        kind = self.kind_var.get()
        try:
            if self.selected_note_id is None:
                self.selected_note_id = self.database.save_note(
                    self.user.id, content, title=title, kind=kind
                )
            else:
                self.database.update_note(
                    self.user.id, self.selected_note_id, title, content, kind
                )
            self._refresh()
        except Exception as exc:
            LOGGER.exception("Could not save a note.")
            messagebox.showerror("Notes", str(exc) or "Could not save this item.", parent=self.window)

    def _delete_item(self) -> None:
        if self.selected_note_id is None:
            return
        if not messagebox.askyesno("Delete item", "Delete this saved item?", parent=self.window):
            return
        try:
            self.database.delete_note(self.user.id, self.selected_note_id)
            self._new_item()
            self._refresh()
        except Exception:
            LOGGER.exception("Could not delete a note.")
            messagebox.showerror("Notes", "Could not delete this item.", parent=self.window)


def main() -> None:
    database = ChatDatabase(DATABASE_FILE)
    try:
        database.initialize()
    except sqlite3.Error as exc:
        messagebox.showerror("Database error", f"Could not initialize the local database:\n{exc}")
        return

    kernel = initialize_aiml_kernel()
    service = ChatService(kernel)
    root = tk.Tk()
    ChatbotApp(root, database, service)
    root.mainloop()


if __name__ == "__main__":
    main()
