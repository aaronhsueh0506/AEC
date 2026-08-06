# Historical Python/C parity harnesses

These unsupported harnesses replay fp64 Python goldens from a retired parity
campaign. Several no longer compile or pass against the current API, so they
are archival reference material—not tests—and must be excluded from release
source packages. Current C regression targets are listed by `make help`;
maintained parity tests remain in the parent `test/` directory.
