# Chatbot Product Requirements

## 1. Product Overview

This project is a local Python desktop chatbot. It combines AIML rules for known questions with a small TF-IDF and Naive Bayes intent classifier for everyday conversation. A Tkinter interface provides account sign-in, chat, and note management.

## 2. Product Goals

- Provide immediate, polite responses to common questions and casual conversation.
- Give users a simple desktop interface with readable chat history and clear message colors.
- Keep account data, saved notes, preferences, and conversation history separated by user in SQLite.
- Keep conversation rules and fallback training examples easy to extend.

## 3. Target Users

- People who want a small local chatbot for common questions and informal conversation.
- Developers adapting the AIML patterns or intent examples for a focused use case.

## 4. Functional Requirements

### Authentication and storage

- Users can create an account and sign in with a username and password.
- The database hashes passwords and stores hashed session tokens with expiry and revocation.
- Users can create, update, and delete notes and preferences.
- Each chat turn is saved to local history and restored at sign-in.

### Conversation

- AIML answers supported greetings, identity questions, basic AI questions, weather inquiries, and other defined patterns.
- Unmatched messages are passed to an intent classifier with conversational examples for greetings, wellbeing, moods, thanks, farewells, jokes, agreement, disagreement, apologies, compliments, and small talk.
- The classifier uses TF-IDF features, Naive Bayes probabilities, and cosine similarity. A small pure-Python similarity fallback keeps responses available if scikit-learn cannot load.
- The fallback corpus is hand-authored starter data. The chatbot does not automatically learn from saved conversations.

### User interface

- The Tkinter app provides sign-in and account creation, a scrollable chat log, message entry, Send, Notes, and Sign out controls.
- Enter sends a message. The entry clears after sending and the chat log scrolls to the latest turn.
- The interface uses a coordinated navy, blue, pale-blue, and pale-green palette with 12-point sans-serif text.

## 5. Scope and Limitations

- The app runs locally for one signed-in user at a time. It is not a multi-user hosted service.
- Weather responses are informational only; no live weather API is connected.
- The conversation model is intentionally small and may misunderstand wording outside its examples.
- Future extensions may add service integrations, more AIML files, training data, and alternate interfaces.

## 6. Project Files

`main.py` starts the app. `database.py` handles SQLite, `ml_fallback.py` contains the intent examples and classifier, `std-startup.xml` loads `basic_chat.aiml`, and `requirements.txt` lists external packages. `README.md` covers setup and use, while `.gitignore` excludes local runtime data. The app creates `bot_brain.brn` and `chatbot.db` beside the source files when it runs.
