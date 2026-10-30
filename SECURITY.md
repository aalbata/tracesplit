# Security boundary

TraceSplit is a local research prototype, not a hardened service. Its language does not evaluate arbitrary Python. CSV resolution is restricted to the selected workspace. Program size, input size, candidate count, one-hot expansion, and some model parameters are bounded. The browser server binds only to 127.0.0.1, requires a per-run token, accepts one run at a time, and applies a subprocess timeout.

These measures reduce accidental misuse. They are not a formal security audit, operating-system isolation, or protection against a hostile local user with Python or filesystem access. Do not expose the server through a reverse proxy or public port. Evidence files include row indices, labels, predictions, and preprocessing statistics; publishing them can disclose information about the input data.

Unkeyed SHA-256 fingerprints provide consistency checks, not authenticity or digital signatures. A party able to rewrite evidence can rehash fabricated records. Numerical replay and counterfactual tests rest on different assumptions, and neither establishes correctness against an arbitrary malicious backend. The injected-fault study in the results folder measures this boundary.

Address reports to the corresponding author of the accompanying article.
