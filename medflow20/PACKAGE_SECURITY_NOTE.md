# Package Security Note

The downloadable **SAFE** project package intentionally excludes local `.env` files so a real API key is not redistributed or accidentally committed.

To run the project locally:

1. Copy `.env.example` to `.env`.
2. Add your current Groq API key locally.
3. Keep `.env` untracked; `.gitignore` already excludes it.
4. Because an API key was previously shared during project work, rotating that key before final submission is recommended.

No source PDF, evaluation label, Chroma database file, or frozen Day 2 benchmark file is removed by this security packaging step.
