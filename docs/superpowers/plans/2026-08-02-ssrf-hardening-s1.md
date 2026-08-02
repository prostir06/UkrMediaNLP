# Sprint 1: SSRF Hardening

**Goal:** Close P0 allowlist public-suffix leak and validate redirects before each hop.

**Done when:** `evil.com.ua` blocked; redirect chains to disallowed hosts never connect; tests green.

## Tasks
1. Replace `parts[-2:]` with multi-part TLD-aware parent domain helper + never add bare `com.ua`.
2. HTTP GET with `allow_redirects=False` + hop loop + `is_allowed_url` per Location.
3. Adversarial + redirect unit tests; ruff; commit.
