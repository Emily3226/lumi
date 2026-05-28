# lumi

## General chat fallback

The general agent answers from `data/general_knowledge.md` first.

If the file does not contain an answer, the app falls back to Groq only.

Put `GROQ_API_KEY=your_key_here` in a `.env` file in the repo root, or set it as a Windows environment variable.