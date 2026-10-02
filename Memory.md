# Chatbot Project Memory

## 1. Current Status

- **Phase:** MVP implementation complete; local startup and GUI review remain.
- **Health:** Ready for a manual first run.
- **Scope:** Local Tkinter application using AIML rules, a small conversational intent classifier, and SQLite user data.

## 2. Implemented

- AIML kernel startup from `std-startup.xml`, with compiled brain loading and rebuild fallback in `main.py`.
- AIML patterns for greetings, casual exchanges, basic AI questions, identity, weather inquiries, and a classifier fallback marker.
- Expanded ML examples and responses for greetings, wellbeing, positive and negative moods, thanks, farewells, jokes, apologies, agreement, disagreement, compliments, and small talk.
- Optional pure-Python TF-IDF/cosine fallback if scikit-learn is unavailable.
- SQLite registration and sign-in, hashed passwords and session tokens, per-user notes/preferences, and conversation history.
- Tkinter sign-in, chat, and notes views with a navy, blue, pale-blue, and pale-green color palette.
- Updated PRD, Architecture, Rules, Design, and Task documents to match the implementation.
- Added a GitHub-ready `README.md` and `.gitignore`; the ignore rules exclude the local database and compiled brain.

## 3. Remaining Work

- Create a local Python environment and install `requirements.txt`.
- Launch `main.py` and check the sign-in, chat, notes, and sign-out flow on the target machine.
- Tune the small hand-authored intent corpus if real messages are misclassified.

## 4. Known Limits

- The classifier uses a small fixed corpus and does not learn automatically from conversation history.
- Weather answers do not use a live weather service.
- The `aiml.py` package requires Python 3.10 or newer. If AIML cannot load, the classifier remains available as a graceful fallback.
- The application is a local desktop MVP, not a service designed for thousands of simultaneous users.

## 5. Validation

- The current conversation and UI edits still need a syntax/XML structural check.
- The GUI and installed third-party package startup still need a manual run in the user's Python environment.
