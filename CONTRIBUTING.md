# Contributing

Create a minimal reproducer and an automated test for each change. Keep the language meaning explicit, reject unsupported behavior instead of guessing, and preserve source locations. Run the full tests, the smoke test, and the research stages after changes to semantics, preprocessing, splitting, or export. Numerical claims are regenerated from saved outputs by the scripts in `scripts/`; do not edit result CSV files by hand.

Changes that enlarge the trusted surface, allow arbitrary Python, or introduce external services need a separate design and security review.
