# Python AIML Chatbot

A small local desktop chatbot built with Python, AIML, scikit-learn, SQLite, and Tkinter. AIML handles known patterns, while a TF-IDF and Naive Bayes intent classifier responds to everyday messages that do not match a rule.

## Features

- Sign up and sign in with a username and password.
- Store accounts, sessions, notes, preferences, and conversation history in a local SQLite database.
- Use AIML patterns for greetings, casual exchanges, AI questions, identity questions, weather questions, and jokes.
- Use a small intent classifier for greetings, wellbeing, moods, thanks, farewells, jokes, apologies, agreement, disagreement, compliments, and small talk.
- Keep chat history visible with distinct user and bot message colors.
- Reuse a compiled AIML brain on later launches for faster startup.

## Requirements

- Python 3.10 or newer.
- Tkinter, included with many standard Python distributions. On Linux, install the distribution's Tkinter package if it is missing.
- Packages listed in `requirements.txt`.

The `aiml.py` package is imported by the application as `aiml`.

## Install and Run

Open a terminal in this project folder. On Windows, create a virtual environment, install dependencies into it, and start the app with that same environment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

If the `py` launcher is unavailable, use the command for a Python 3.10+ interpreter installed on your machine. On macOS or Linux, use:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On first launch, select **Create a new account** and enter a username of 3–32 characters with no spaces and a password of at least 8 characters. Sign in with those credentials on later launches.

## Project Layout

```text
Chatbot/
├── main.py                 # Application entry point, AIML setup, routing, and Tkinter UI
├── database.py             # SQLite accounts, sessions, notes, preferences, and history
├── ml_fallback.py          # Intent examples, TF-IDF/Naive Bayes classifier, and replies
├── std-startup.xml         # AIML startup rule that loads basic_chat.aiml
├── basic_chat.aiml         # Explicit conversational patterns and fallback marker
├── requirements.txt        # Third-party Python packages
├── README.md               # Setup and project guide
├── PRD.md                  # Product requirements
├── Architecture.md         # Application components and data flow
├── Rules.md                # Development guidance
├── Design.md               # Interface and color palette
├── Memory.md               # Current implementation status and known limits
└── Task.md                 # Setup and first-run checklist
```

`bot_brain.brn` and `chatbot.db` are generated beside the source files the first time the app runs. They are local runtime data and are excluded from version control.

## How Conversation Routing Works

1. `main.py` checks for the compiled AIML brain and attempts to load it.
2. If the brain is missing or cannot be loaded, the kernel reads `std-startup.xml`, loads `basic_chat.aiml`, and saves a compiled brain when supported.
3. The chat service checks AIML rules first. Its default AIML pattern marks unmatched messages for the ML fallback.
4. `ml_fallback.py` compares the message against hand-authored training examples using TF-IDF, Naive Bayes probabilities, and cosine similarity. If scikit-learn cannot be imported, a pure-Python TF-IDF cosine fallback is used.
5. Each signed-in user's conversation turns are stored in SQLite and restored in the chat log at sign-in.

## Extending the Chatbot

- Add exact phrase rules and scripted replies to `basic_chat.aiml`.
- Add varied example phrases and a response intent to `ml_fallback.py` for conversational wording that may take many forms.
- Keep the AIML files beside `main.py`; the startup rule loads `basic_chat.aiml` by its filename.

The classifier uses a small fixed corpus. It does not train itself from saved conversation history. Weather replies do not use a live weather service, and the app may not understand wording outside its rules and examples.

## License

No license file has been added yet.
