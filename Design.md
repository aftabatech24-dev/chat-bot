# Chatbot Interface Design

## 1. Design Principles

- Keep the interface calm, familiar, and easy to scan.
- Make the newest chat activity visible automatically.
- Distinguish user and bot messages with soft color fills and readable text contrast.
- Keep account, chat, and notes controls visually consistent.

## 2. Color Palette

| Use | Color | Value |
| --- | --- | --- |
| App background | Pale blue-gray | `#EEF3F9` |
| Header | Deep navy | `#1E3A5F` |
| Primary buttons | Blue | `#2563EB` |
| Button active state | Darker blue | `#1D4ED8` |
| Chat log | Near-white | `#F8FAFC` |
| Input area | Light gray-blue | `#F1F5F9` |
| User message | Pale blue | `#DBEAFE` |
| Bot message | Pale green | `#DCFCE7` |
| Main text | Charcoal | `#1F2937` |

## 3. Typography

- Use Arial or the operating system's sans-serif font.
- Use 12-point text for chat and controls, with a larger bold welcome heading.
- Use dark text on the pale message backgrounds for contrast.

## 4. Interface Components

- **Sign-in screen:** username and password fields with an account-creation option.
- **Chat header:** signed-in username, Notes, and Sign out controls on a navy background.
- **Chat history:** scrollable, read-only `Text` widget with pale-blue user messages aligned right and pale-green bot messages aligned left.
- **Message composer:** light input area with a single-line entry and blue Send button. Enter sends the message and clears the entry.
- **Notes window:** list of saved notes and preferences with title, type, and editable content fields.
