# Chatbot Project Tasks

## Phase 1: Project Setup

- [x] Create the modular project files and dependency list.
- [x] Add a project README and ignore local runtime data.
- [x] Add AIML startup routing through `std-startup.xml`.
- [ ] Create a local virtual environment and install `requirements.txt`.

## Phase 2: Authentication and Account Data

- [x] Add username registration and password sign-in.
- [x] Generate expiring session tokens and store only their hashes.
- [x] Scope notes, preferences, and conversation history to the signed-in user.

## Phase 3: Conversation Engine

- [x] Add AIML rules for common questions and casual exchanges.
- [x] Load an existing `bot_brain.brn` or learn the startup AIML files and save a compiled brain.
- [x] Add conversational intent examples and responses for the ML fallback.
- [x] Add a low-confidence response and a pure-Python similarity fallback.
- [ ] Review intent predictions against real user phrases and tune the examples as needed.

## Phase 4: Interface and Notes

- [x] Build the Tkinter sign-in, chat, and notes views.
- [x] Add a color palette with distinct user and bot message backgrounds.
- [x] Add Enter-to-send, input clearing, scroll-to-latest, and history restoration.
- [x] Add note and preference creation, editing, and deletion.

## Phase 5: First-Run Review

- [ ] Install the dependencies in the local virtual environment.
- [ ] Launch `main.py` and create a test account.
- [ ] Review AIML answers, everyday conversation, notes, history, and sign-out.
- [ ] Record any platform-specific AIML or Tkinter setup adjustments.
