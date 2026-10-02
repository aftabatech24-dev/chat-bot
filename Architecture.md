# Chatbot System Architecture

## 1. Components

- **Tkinter interface (`main.py`)** provides authentication, chat, saved notes, and conversation history.
- **Chat service (`main.py`)** sends each message to AIML first and routes the AIML fallback marker to the intent classifier.
- **AIML kernel (`aiml.Kernel`)** loads a compiled brain when available or parses `std-startup.xml` and `basic_chat.aiml` on first launch.
- **Intent classifier (`ml_fallback.py`)** classifies messages outside explicit AIML patterns using TF-IDF, Naive Bayes, and cosine similarity. If scikit-learn is unavailable, it uses a small TF-IDF cosine implementation.
- **SQLite storage (`database.py`)** manages accounts, expiring sessions, notes, preferences, and per-user chat history.

## 2. Message and Startup Flow

1. The app initializes the SQLite schema and loads the AIML kernel.
2. If `bot_brain.brn` exists, the kernel attempts to load it. If loading fails or the file is missing, the kernel learns `std-startup.xml`, which loads `basic_chat.aiml`, then saves a compiled brain when supported.
3. After sign-in, the app restores the user's recent conversation history.
4. Each message is matched against AIML patterns. A direct AIML response is shown immediately; the default marker routes unmatched text to `ml_fallback.py`.
5. The user message and response are written to the authenticated user's SQLite history.

## 3. Data and Security

- Passwords are stored as salted PBKDF2-HMAC-SHA256 digests, not plaintext.
- Session tokens are random opaque values. SQLite stores only their SHA-256 hashes, expiry times, and revocation timestamps.
- Notes, preferences, and conversation turns reference a user ID and cascade-delete with that account.
- SQLite is local to the project folder by default; this desktop architecture is not designed for thousands of concurrent users.

## 4. Technology

- Python 3.10 or newer
- AIML through `aiml.py` (the package is imported as `aiml`; it targets Python 3.10 and newer)
- scikit-learn and NumPy for the primary classifier
- SQLite through Python's standard library
- Tkinter for the desktop interface

## 5. Folder Structure

```text
Chatbot/
├── main.py                 # Application entry point, service routing, and Tkinter UI
├── database.py             # Account, session, notes, preferences, and history storage
├── ml_fallback.py          # Training examples, intent classifier, and responses
├── std-startup.xml         # AIML startup rule that learns basic_chat.aiml
├── basic_chat.aiml         # Explicit conversation rules and ML fallback marker
├── requirements.txt        # External Python packages
├── README.md               # Setup and usage guide
├── .gitignore              # Excludes local environment and private runtime data
├── PRD.md                  # Product requirements
├── Architecture.md         # Component and data flow
├── Rules.md                # Development rules
├── Design.md               # Interface and color specification
├── Task.md                 # Implementation and run checklist
├── Memory.md               # Current project status
├── bot_brain.brn           # Generated compiled AIML brain
└── chatbot.db              # Generated local SQLite database
```

`bot_brain.brn` and `chatbot.db` are created when the app runs; they do not need to be authored by hand.
