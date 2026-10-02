# Chatbot Development Rules

## 1. Separation of Responsibilities

- Put account, session, note, preference, and history persistence in `database.py`.
- Put AIML loading, message routing, and Tkinter presentation in `main.py`.
- Put intent examples, classifier behavior, and fallback responses in `ml_fallback.py`.
- Put deterministic phrase matching and scripted answers in the AIML files.
- Keep UI code independent of SQLite details by calling the database class methods.

## 2. Updating Conversation Behavior

- Add exact, reliable question-and-answer patterns to `basic_chat.aiml`.
- Add varied example phrasings and suitable intent responses to `ml_fallback.py` for casual language that should work across multiple wordings.
- Keep training examples representative and focused. Do not claim that this small classifier learns from user chats; saved history is not training data.
- Use a helpful, honest response for low-confidence messages. Do not invent live data or capabilities that are not connected.

## 3. Python and Data Practices

- Favor readable functions, type hints, and context managers for database and file resources.
- Use parameterized SQL queries and validate inputs before persistence.
- Store password digests and session-token hashes, never plaintext credentials or tokens.
- Keep database operations scoped to the authenticated user's ID.
- Use the `aiml.py` distribution for current Python versions; Python imports it as `aiml`.
- Handle missing or incompatible AIML/scikit-learn dependencies gracefully where the existing fallback allows it.

## 4. Brain and File Layout

- Check for `bot_brain.brn` with `os.path.isfile` before learning AIML startup files.
- If the brain is absent or invalid, load `std-startup.xml`, compile its AIML rules, and save a new brain when the library supports it.
- Keep `basic_chat.aiml` beside `std-startup.xml` because the startup rule loads it by filename.
- Do not commit or hand-edit runtime-generated `bot_brain.brn` or `chatbot.db` files as source.
