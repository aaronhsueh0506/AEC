# Changelog

All notable changes to this AEC implementation. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) but adapted for the
research-arc workflow used here.

**Retired evidence documents.** Entries below `[4.0.0]` cite per-arc verdict and
closure documents under `docs/` (`docs/f2_4_verdict.md`, `docs/v3_14_plan.md`,
and ~69 others). Those files were removed in the 4.0.0 release cleanup: they are
development history, not release surface, and shipping them to an integrator
would present superseded conclusions as current specifications. **They are not
missing — every one is recoverable from git history** at the commit the entry
describes, e.g. `git log --all --diff-filter=D -- docs/f2_4_verdict.md` to find
the deleting commit, then `git show <sha>^:docs/f2_4_verdict.md`. Treat every
`docs/*.md` path in an entry below `[4.0.0]` as a git-history reference, not a
live file. The `[4.0.0]` entry itself cites no `docs/` path — its evidence is
checked in under `eval/ab_evidence/` and `docs/timing_constant_inventory.md`.
List the retired ones with (this reports the two example paths above as well,
since they are themselves retired documents):

```bash
grep -oE '\bdocs/[A-Za-z0-9_/-]+\.(md|html)' CHANGELOG.md \
  | while read -r p; do [ -f "$p" ] || echo "retired: $p"; done
```

Versioning: `__version__` in [python/aec.py](python/aec.py) versions this
library's PUBLIC API and OUTPUT CONTRACT — the C header
`c_impl/include/aec.h` (struct layouts, entry-point signatures, context-field
semantics), the auto-derived default signal grid, the `--debug-trace` CSV
schema, and whether `aec_process()` output stays bit-identical. MAJOR bumps
when any of those break; MINOR when production BALANCED behaviour changes
compatibly or new surface is added; PATCH for fixes that keep both the API and
the output contract intact. Through `[3.24.1]` this field instead tracked only
the production-graded BALANCED preset, with `v3.x.y` jumping per production
change into BALANCED and `v3.x` arc closure bumping `x` — entries below
`[4.0.0]` were written under that older rule and are not restated here.

Bench standard for every entry below: 800-case AEC Challenge corpus,
`preset=balanced / fl=832 (52 ms) / cng=True / -j 4`. Listen evidence cited
when verdict requires it.

---

## [Unreleased] — 2026-08-16 — delay productization line A, step 4 (CLI + memory diagnostics + docs/ABI) — A-line complete

### Hardening and consistency fixes

- Low-level `delay_aec3_get_mem_size()`/`delay_aec3_init()` now REJECT
  out-of-range `num_filters` (0 / nonzero error) instead of silently
  clamping to [1, 5]; `LegacyDelayShim` direct construction enforces the
  same range. Fail-fast now holds at every public layer.
- Doc reconciliation: STATIC_MEMORY.md pool figures caught up to the
  +16 B census shift; aec.html's seam-state list corrected
  (`EXTERNAL_ALIGNED` is LOCKED from hop 0, not UNLOCKED); the user
  manual's `delay_num_filters` row no longer conflates small-bank
  residual tracking with `FIXED` (which builds no matched filter at all).


Plan §9.4 (last A-line step before the AEC push gate). Steps 1-3 (three-state
`delay_mode`, the shared signal-grid resolver, and the pool-first
`DelayAec3`/per-mode ring) landed the mechanism; this step makes it operable
and measurable from the CLI, and syncs every downstream document. **Output
contract intact**: byte-exact vs `551e70d` (pre-step-4 HEAD) on 2 spot-check
cases, SHA-256-compared — see "Byte-exact evidence" below.

### Added

1. **`aec_wav` CLI**: `--delay-mode {matched|fixed|external}`,
   `--delay-num-filters <n>`, `--fixed-delay <samples>`, `--print-mem-size`.
   All three delay-mode flags are handed to `AecConfig` **unvalidated**, same
   convention as every other CLI override in this file —
   `aec_validate_config()` is the single range/compatibility authority, so an
   illegal value or mode/field combination fails at `aec_create()` (or, for
   `--print-mem-size`, at `aec_get_mem_breakdown()`) with a fail-fast error
   listing every delay field, never a silent clamp. `--print-mem-size` prints
   one `mem: sr=... fft=... hop=... mode=... n=... fixed_delay_samples=...
   total_bytes=... estimator_bytes=... ring_bytes=...` line to **stdout** and
   exits before opening a WAV writer or running a single hop — no audio is
   touched, matching plan §3.4.6 RAM acceptance test 6's mono `--print-mem-size`
   requirement.
2. **`AecMemBreakdown` + `aec_get_mem_breakdown()`** (`aec.h`/`aec.c`): labels
   two subsets of the `aec_get_mem_size()` total instead of making a caller
   re-derive them — `total_bytes` is that exact call (never a second
   computation of it), `estimator_bytes` is `delay_aec3_get_mem_size()` for the
   resolved `(sample_rate, hop, delay_num_filters)` triple (0 outside
   `MATCHED`), `ring_bytes` is the mode-aware alignment-ring size (0 for
   `EXTERNAL_ALIGNED`). Backs `--print-mem-size` and is documented at
   `c_user_manual_zh_TW.md` §6.7.
3. **Matched-filter duty-cycle engagement census** (`duty_hops_total` /
   `duty_hops_run` on `Aec`, surfaced via `AecDebugStatus`): the duty machine's
   doc comment in `aec.h` claims it cuts "~90% of the matched-filter cost" as a
   DESIGN figure that was never measured — on an echo path whose estimate
   keeps moving the machine re-arms every time and never decimates at all, so
   the realised saving can be nothing like the design number. The two counters
   make it measurable per run: incremented at the single call site that
   consumes `run_filter` (so the census cannot drift from what actually
   executed), reset with `aec_reset()` (a ratio spanning two duty regimes means
   nothing), and only accumulate under `AEC_DELAY_MATCHED` (the only mode that
   ever builds an estimator — `FIXED` and `EXTERNAL_ALIGNED` both leave
   `has_delay == 0`, so both leave the census at 0 for the same structural
   reason, not two different ones). `example/aec_wav.c` prints one
   `duty: matched_filter_ran=R/T hops (E% engagement, S% saved)` stderr line
   per run (measured on the 10 s reference pair: **379/1250 = 30.32%
   engagement** — reproduces the pre-existing research-branch measurement of
   the same underlying mechanism exactly). Diagnostic only: two counter
   increments off the signal path, adding exactly 16 B to `sizeof(Aec)` (see
   ABI below) and changing no output sample (byte-exact evidence below).
4. **Tests**:
   - `test/test_duty_census.c` (9 checks): `duty_hops_total` == hops
     processed under `MATCHED`; `duty_hops_run < duty_hops_total` on a stable
     3 s single-tap echo (the assertion that actually distinguishes a real
     census from one wired to the hop counter); engagement lands in the
     decimated band (0%, 60%); `FIXED` and `EXTERNAL_ALIGNED` both leave the
     census at 0; `aec_reset()` zeroes it and counting resumes afterward.
     Mutation-verified: dropping the `run_filter` guard (counting every hop as
     "run") fails 2/9; dropping the `aec_reset()` zeroing fails 2/9. Both
     restored and re-run green.
   - `test/cli_delay_flags.sh` (43 checks, mirrors `test_delay_num_filters.c`'s
     "CLI plumbing only" scope — the bank-size geometry, ring-size formula, and
     mode×field validation matrix are already covered by the C regression
     suites and are not re-tested here): every new flag reaches `AecConfig`;
     every illegal mode/field combination is rejected, including
     **`--fixed-delay` negative under `FIXED`** (the plan's requested mutation
     target — mutation-verified: silently clamping a negative value to 0
     *before* the `cfg` assignment, instead of the CLI's documented
     hand-through-unvalidated contract, turns this and two downstream checks
     red; restored and re-run green); a rejected run writes no output; the
     default run is byte-identical to explicit
     `--delay-mode matched --delay-num-filters 5`; `--print-mem-size` touches
     no audio and its `total_bytes`/`estimator_bytes`/`ring_bytes` are
     cross-checked against **`test/print_mem_size_ref.c`**, a tool that calls
     `aec_get_mem_breakdown()` from its own source file specifically so the
     check exercises a code path independent of `aec_wav.c`'s printf
     (mutation-verified: `+16` on the printed `total_bytes` in `aec_wav.c`
     turns all 5 mem-case checks red; restored and re-run green).

### Changed

5. **ABI: `sizeof(Aec)` +16 B, every documented pool-size figure shifts by
   exactly +16 B.** The two new `unsigned long long` census fields are
   appended at the end of the `Aec` struct (existing field offsets unchanged).
   Because `aec_get_mem_size()` adds `ALIGN16(sizeof(Aec))` as one flat,
   mode/n/grid-independent term, this is a **uniform +16 B shift on every
   absolute total** (confirmed empirically on all 4 grids × `MATCHED` n=1..5 ×
   `FIXED`/`EXTERNAL_ALIGNED` via `make test-delay-num-filters`'s printed
   table) and **zero change to every delta** (the 5,728 B/filter step, the
   `FIXED` byte/ms ring formula, and every mode-to-mode difference in
   `c_user_manual_zh_TW.md` §4 are unaffected — both sides of a subtraction
   carry the same +16 and it cancels). Representative before/after (KISS,
   balanced, 16 kHz/256): `MATCHED` n=5 379,696 → **379,712 B**; `MATCHED` n=1
   356,784 → **356,800 B**; `EXTERNAL_ALIGNED` 214,688 → **214,704 B**.
   **A caller that hardcoded a byte count, or skipped the "any field change
   requires re-querying `aec_get_mem_size()`" rule from §2.4/§3.3, must
   rebuild and re-query.** No source change is required for callers who
   already use `aec_get_mem_size()`/`aec_get_mem_breakdown()` (the normal
   path). This repo carries no C-side `AEC_VERSION`/`AEC_ABI`/layout-version
   macro (`grep AEC_VERSION\|AEC_LAYOUT` over `c_impl/include`+`c_impl/src`
   returns nothing) — `AecConfig`/`Aec` layout changes are tracked here in the
   CHANGELOG only, per the productization plan's fallback rule (§3.3: "bump
   AEC ABI/layout/descriptor version; update changelog and rebuild-required
   note" — bump if a version macro exists, changelog-only otherwise).
   `python/aec.py`'s `__version__` stays `4.0.0rc1` for this commit; bumping
   it to `4.0.0rc2` (or `4.0.0`) is a **release-time** decision the plan defers
   to the actual push/tag, not to each intermediate productization step.

### Docs

6. **`c_user_manual_zh_TW.md`**: new §6.7 (`AecMemBreakdown`/
   `aec_get_mem_breakdown()`), new §8.5 (duty-cycle census contract + test
   pointer), Appendix A's flag table gets the four new CLI flags plus a
   `--print-mem-size` output subsection and a "Duty census 診斷輸出"
   subsection (both with a real captured example line). Every absolute
   pool-size figure in §4's tables (the grid×n table, the delay_mode
   comparison table, the FIXED-per-grid table, the 2500 ms/48 kHz note, the
   `enable_shadow=0` entry) is updated +16 B with a note explaining the shift
   is uniform and delta-preserving (identical reasoning to item 5 above).
7. **`docs/html/aec.html`**: was stale since BEFORE step 1 (still carried
   pre-refactor pool-size figures and, more seriously, a `delay_num_filters`
   paragraph that said "RAM 不變(C 端陣列維持編譯期上限)" — flatly wrong
   after step 2/3's pool-first refactor). Brought current: summary mem-size
   table corrected to the real totals; new "Delay mode" subsection (3-mode
   table + a `MATCHED`/`FIXED`/`EXTERNAL_ALIGNED` memory comparison table,
   both cited from `c_user_manual_zh_TW.md` §4/§2.1, not invented); the
   `delay_num_filters` paragraph corrected to state RAM DOES scale
   (5,728 B/filter, all four grids); `aec_get_mem_breakdown()` documented;
   the `ref_ring` row in the State table corrected from the retired
   `enable_delay_est=0` framing to per-`delay_mode`; the `AecLinearContext`
   `UNLOCKED` bullet and the `aec_debug_status()` paragraph updated for
   `delay_mode`/the duty census; the CLI flags and both new test targets
   added to the file list. Still self-contained, no external references,
   24,109 B (< 30 KB budget).

### Byte-exact evidence

Verified via an isolated `git worktree` at `551e70d` (this step's parent
commit, pre-CLI/pre-census), built KISS-backend, and SHA-256-compared against
this commit's build on the same 10 s reference pair
(`wav/aec_record/aec_record_{mic,ref}_10s.wav`):

| case | flags | SHA-256 match |
|---|---|---|
| default, `MATCHED` n=5, 16k/256 | `--cng` | identical (`32ae3b34...`) |
| `MATCHED` n=5, alternate 16k/512 grid | `--cng --fft-size 512` | identical (`c4fc34ef...`) |

Both cases exercise the census-incrementing code path (`has_delay == 1`) on
two different hop counts/K values. The only difference in `aec_wav`'s own
stderr output is the new summary fields and the new `duty:` line — the WAV
bytes themselves are untouched.

### Full regression

`pytest python/tests`: 227 passed (unchanged — no Python touched this step).
C, both backends (KISS, NE10), every wired test target green: `test-counter-
saturation` (12,023 checks), `test-rate-structural` (360), `test-config-
validation` (388), `test-process-context` (70), `test-linear-context` (188),
`test-delay-num-filters` (166), `test-duty-census` (9, **new**),
`test-cli-delay-flags` (43 checks, **new**), `test-shared-far-spec` (51),
`test-shared-fft-handle` (63), `test-zero-heap` (50 cycles, 0 allocator
calls), `test-delay-reset` (16), `test-detectors-parity`/`-all` (0/260 diffs
on both 16 kHz and legacy 48 kHz), `test-static-aec` (669,824 samples
byte-equal, pool 379,712 B KISS / 379,104 B NE10 @16k/256).

---

## [Unreleased] — 2026-08-16 — delay productization line A, step 3: per-mode alignment ring

Plan §3.2.8-9, §9.3. Step 2 stopped constructing an ESTIMATOR outside
`MATCHED`; the reference alignment RING was still sized identically for every
mode that had one. It now follows the mode. **Pool layout changes for
`FIXED` — re-query `aec_get_mem_size()`** (no source change; `MATCHED` and
`EXTERNAL_ALIGNED` pool sizes are unchanged). **Output contract intact**:
byte-exact vs `5241d99` on both ports across
{16k/256, 16k/512, 48k/1024} × {`MATCHED` n=5, `MATCHED` n=1, `FIXED` at
0/100/128/1536/1600/8137/40000, `EXTERNAL_ALIGNED`} — 15 C cases (FNV-64 over
every output hop) and 15 Python cases (SHA-256 over the concatenated output),
all identical.

### Changed

1. **The reference ring is sized per mode, by ONE checked helper** —
   C `aec_ref_ring_samples()` (`aec.c`), Python `ref_ring_samples()`
   (`modules/config.py`), term for term the same:

   | mode | ring capacity (samples) |
   |---|---|
   | `MATCHED` | `max(delay_buffer_ms, max_delay_ms + 4096)` — unchanged |
   | `FIXED` | `fixed_delay_samples + hop_size` |
   | `EXTERNAL_ALIGNED` | 0 |

   `aec_get_mem_size()` and the `aec_carve()` pool walk take the number from
   the same `aec_derive_dims()` call, so the query and the carve cannot
   describe different rings. `MATCHED` keeps the legacy `+4096` headroom
   deliberately: that ring size ALSO gates which estimates the controller may
   accept (`new_delay <= ref_ring_size - hop`), so it is behaviour, not slack.

2. **`FIXED` no longer borrows the search ring.** `max_delay_ms` and
   `delay_buffer_ms` are now inert in that mode — a config differing only in
   those two fields produces a byte-identical pool. The `fixed + hop` bound
   is TIGHT, not an estimate: the ring holds absolute samples `[T-rs, T)`,
   each hop writes the newest `hop` (advancing `T`) and then reads
   `[T-d-hop, T-d)`, so validity is exactly `rs >= d + hop`; under `FIXED`,
   `d` is immutable (no estimator, and `reset()` re-seeds the same value), so
   this bounds a constant rather than a moving quantity and no headroom term
   is meaningful. Equality is safe — at `rs == d + hop` the read starts on
   the write cursor, i.e. the one hop this hop's write did not overwrite.
   `fixed_delay_samples = 0` degenerates to a hop-sized, write-only ring
   (the read is skipped by the `current_delay > 0` guard), kept rather than
   special-cased to NULL so `FIXED` has exactly one ring code path.

3. **`aec_get_mem_size(cfg)` now scales with `fixed_delay_samples`** (KISS,
   balanced, 16 kHz/256), where every `FIXED` config used to return the same
   345,760 B:

   | config | pool | vs `MATCHED` n=5 |
   |---|---:|---:|
   | `MATCHED` n=5 (default) | 379,696 B | — |
   | `FIXED` 0 | 215,200 B | −164,496 B |
   | `FIXED` 400 (25 ms) | 216,800 B | −162,896 B |
   | `FIXED` 1600 (100 ms) | 221,600 B | −158,096 B |
   | `FIXED` 8000 (500 ms) | 247,200 B | −132,496 B |
   | `EXTERNAL_ALIGNED` | 214,688 B | −165,008 B |

   A typical fixed-path product (25 ms measured delay) drops from 345,760 B
   to 216,800 B, i.e. **−128,960 B**. Per-grid tables in the manual (§5) and
   printed by `make test-delay-num-filters`. Note the ring must still hold
   the delay: a 2500 ms `FIXED` delay at 48 kHz costs 1,220,512 B, ABOVE
   `MATCHED` n=5 — the memory-efficient answer to a large fixed delay is for
   the caller to compensate it and use `EXTERNAL_ALIGNED`.

### Fixed

4. **`EXTERNAL_ALIGNED` left one delay attribute unset in Python.** The
   orchestrator never assigned `_current_delay` in that branch, and
   `get_stats()` / the `_diag` walk read it unconditionally — so
   `AEC(delay_mode=EXTERNAL_ALIGNED).get_stats()` raised `AttributeError`
   (present since step 1; audio path unaffected, every other reader is behind
   `_delay_active`). It is now spelled `0`, the contract value, matching the
   C port's `a->current_delay = 0` and what `aec_debug_status()` reports.
   Only visible change beyond the fixed crash: `_diag['delay_samples']` reads
   0 instead of -1 in that mode.

### Added

5. **Tests** — Python `test_delay_mode.py` 22 → 32 cases; C
   `test_linear_context.c` 111 → 188 and `test_delay_num_filters.c`
   109 → 166. New coverage: the ring-size formula per mode on every grid
   (`FIXED` at 0 / below-a-hop / exactly-a-hop / exact-hop-multiple /
   NON-multiple / past-`max_delay_ms`); `max_delay_ms`/`delay_buffer_ms`
   proven inert under `FIXED`; the served far byte-equal to the caller's far
   delayed by exactly `fixed_delay_samples` across hundreds of ring wraps
   (the split two-part read is now the COMMON path — 218/400 hops at
   `fixed=100` — not the rare edge case it was against a 32,768-sample ring),
   the same again after `aec_reset()`'s refill with the RAW-far window
   exactly as long as at init, a synthetic echo actually cancelled at the
   right fixed delay (34.3 dB) and not at a wrong one (0.1 dB),
   `EXTERNAL_ALIGNED` far passthrough sample-exact, and its pool proven equal
   to `FIXED(0)`'s minus exactly one hop-sized array — the concrete meaning
   of "carries no delay bytes".

### Notes

- **Mutation-checked, red/green both recorded** (4 C + 3 Python): dropping
  `+ hop` from the `FIXED` arm → C `test_linear_context` faults and
  `test_delay_num_filters` 38 red, Python 5 red; shifting the ring read by
  one sample → C 15 red, Python 4 red (`test_delay_num_filters` stays green,
  correctly — it sizes, it does not read); letting `EXTERNAL_ALIGNED` fall
  through to a ring → C 3 red on the pool-composition row, Python 1 red on
  the helper; additionally wiring that ring into the carve → C 8 red. Every
  mutation restored and re-run green.
- **The ERLE correctness test runs the shipped chain (RES on) on purpose.**
  `enable_res=False` is not a neutral simplification here: it also starves
  the AEC3 ERLE feed (`last_erle_windowed` is cached only under
  `enable_res or return_res_context`), which stalls that configuration at
  ~3 dB in EVERY delay mode alike — verified identical on `5241d99`, so it is
  a property of that diagnostic config, not of the alignment path.
- Plan §3.4.6's `--print-mem-size` diagnostics (resolved grid / mode / n /
  estimator bytes / ring bytes) remain step 4.

## [Unreleased] — 2026-08-16 — delay productization line A, step 2: pool-first, config-sized `DelayAec3`

Plan §3, §9.2. **`sizeof(Aec)` and the pool layout change — a rebuild is
required** (no source change for callers who already use
`aec_get_mem_size()`; a caller who hardcoded a byte count must re-query).
**Output contract intact**: `MATCHED` + n=5 renders **sample-exact** to the
pre-refactor baseline `e1142a4` across the full KISS/NE10 × SIMD=0/1 ×
{16k/256, 16k/512, 48k/1024, 8k/256} × {default, CNG, `--no-delay-est`}
matrix — 48/48 cases, compared sample-by-sample rather than by file hash
(float WAV `PEAK` chunks carry a timestamp). The delay chain itself is
additionally pinned bit-exact on its own: the C-regression golden generated
from the **pre-refactor** sources replays through the post-refactor estimator
with 0 mismatches on all four public outputs over 4187 hops.

### Changed

1. **`DelayAec3` is now metadata + pool pointers.** The matched-filter
   coefficient bank, its accumulated-error rows, the down-sampled render
   ring, both lag histograms and their 250-entry time-window rings, the
   single instantaneous prefix-error buffer, and the 48 kHz anti-alias
   sidechain scratch are all carved from the CALLER's pool, sized by the
   resolved `(sample_rate, hop_size, num_filters)` triple. `Aec` no longer
   embeds any of it.

   *This withdraws the previous, deliberate "compute shrinks, RAM does not"
   trade-off* (arrays carved at the 5-filter bound to keep the footprint a
   single compile-time constant). The footprint was already a function of the
   init config — `aec_get_mem_size(cfg)` — so pinning one field of it to a
   constant bought nothing a consumer could use.

2. **`delay_aec3_get_mem_size(sample_rate, hop_size, num_filters)` +
   `delay_aec3_init(d, mem, bytes, sample_rate, hop_size, num_filters)`**
   replace `delay_aec3_init(d, sr)` / `delay_aec3_init_ex(d, sr, n)`. One
   canonical pool-first lifecycle, no construct-from-a-bare-struct entry
   point. Both halves derive every size from ONE internal layout helper and
   clamp `num_filters` through one shared function, so a query and an init
   cannot describe different blocks. `aec_carve()` is the only production
   caller and passes the resolved hop, never a guess.

3. **`aec_get_mem_size(cfg)` shrinks with the delay config** (KISS, balanced,
   16 kHz/256): 379,696 B at the `MATCHED` n=5 default, down to 356,784 B at
   n=1, 345,760 B for `FIXED` and 214,688 B for `EXTERNAL_ALIGNED`. Each
   dropped filter is exactly **5,728 B** on every grid — coefficients 2,048 +
   accumulated error 512 + render ring 1,536 + highest-peak histogram 1,536 +
   pre-echo histogram 96, every term a multiple of 16 so the per-field
   ALIGN16 padding is identical at every n. Full four-grid × n=1..5 table in
   the manual (§5) and printed by `make test-delay-num-filters`.
   *(Non-`MATCHED` modes carving no estimator falls out of this step because
   only `MATCHED` constructs one; the `FIXED` / `EXTERNAL_ALIGNED` alignment
   RING differentiation is still plan step 3.)*

4. **The n=5 default is also 1,376 B smaller than before**, because the two
   48 kHz sidechain scratch buffers are no longer carried at 16 kHz — they
   are carved only when `sample_rate == 48000`, sized `hop/3 + 1` from the
   resolved hop instead of a fixed 192.

5. **`DA_NUM_FILTERS` is demoted to a geometry upper bound** used by
   validation and the reach table. It sizes nothing. The removed
   `DA_RING_CAPACITY` / `DA_MAX_FILTER_LAG` / `DA_HP_HIST_SIZE` /
   `DA_PE_HIST_SIZE` / `DA_RESAMPLE48_SCRATCH_MAX` constants are replaced by
   `DA_*_FOR(n)` / `DA_RESAMPLE48_CAP_FOR(hop)` formula macros, and every
   runtime size is readable off the instance (`render_ring.capacity`,
   `highest_peak.hist_size`, `pre_echo.hist_size`, `resample_cap`).

### Added

6. **Safety, per plan §3.3.** Both walks go through the same checked size
   helpers (`mem_align.h`'s saturating `ck_*`), so a size computation that
   overflows reports failure rather than a wrapped total. `delay_aec3_init()`
   rejects a NULL or non-16-byte-aligned base and an undersized block, and
   **asserts the carve consumed EXACTLY the queried byte count** — not
   "within", exactly — so a drift between the two field walks fails
   construction instead of running on a layout nobody sized. Every carved f32
   array is ALIGN16 (plan §11.7: the NEON matched-filter kernels' 512-tap
   loads). Still zero heap on both construction paths.

7. **Tests** — `test_delay_num_filters.c` 33 → 109 cases: hand-computed pool
   geometry per n (ring 528/912/1296/1680/2064 samples, highest-peak
   897/1281/1665/2049/2433 bins, pre-echo 56/80/104/128/152 bins), the exact
   5,728 B step on all four grids, `FIXED` < `MATCHED` n=1 and
   `EXTERNAL_ALIGNED` < `FIXED` on all four, `aec_create` vs `aec_init`
   sample-exact at every n, the resolved-hop pass-through (the sidechain
   scratch is 171 entries at 48 kHz and absent at every other rate), and the
   pool rejection contract. `test_delay_reset.c` +2: geometry and every pool
   pointer survive a reset.

### Notes

- **The histogram sizing formula deliberately keeps one filter of headroom**
  (`n*SHIFT + SIZE`, not the arithmetically tight `(n-1)*SHIFT + SIZE`). This
  is not caution, and it is not a local invention: it is upstream AEC3's own
  `MatchedFilter::GetMaxFilterLag()`, mirrored verbatim by this repo's Python
  port (`matched_filter.py`'s `get_max_filter_lag()`, whose docstring spells
  the over-count out and states its sizing-only role). Diverging would have
  put the C histogram geometry out of step with both. Independently, the
  pre-echo histogram's bin count is BEHAVIOURAL: its
  windowed local-max scan walks fixed 32-bin windows with a 0.7^k penalty and
  stops when fewer than 32 bins remain, so the bin count decides how many
  windows are scanned. Under the tight formula n=2 would get 56 bins = ONE
  window covering bins 0..31, while that bank can itself report up to bin 55
  — silently discarding half its own search range. Checked for every n in
  [1,5]: the headroom form always scans exactly the windows holding the
  bank's reachable bins, and every window it drops is provably all-zero (it
  sits above the reachable lag) and so can never win the scan. Result: n=1..5
  each keep the behaviour they had when the arrays were carved at the n=5
  bound, and n=5 reproduces the pre-refactor array sizes exactly.
- **Mutation-checked, measured not assumed** (4 mutations, red/green both
  recorded): (a) sizing one field differently in the query walk than in the
  carve → the exact-consumption assertion fires, 4 test targets red
  (delay-num-filters 26 failures, config-validation 45, delay-reset 2,
  static-aec fails to construct); (b) pinning the geometry macros back to the
  n=5 bound → 28 rows red; (c) `aec_carve` ignoring `cfg->delay_num_filters`
  → 20+ rows red (and the get-mem-size-still-honest variant of the same
  mutation segfaults, i.e. the pool really is sized off that field);
  (d) `aec_carve` passing a hardcoded hop instead of the resolved one → the
  grid pass-through row red, and 524,798/576,000 samples of the 48 kHz output
  change. **A first draft of the geometry test was fully green under
  mutation (b)** because it read back the same `DA_*_FOR(n)` macros it was
  checking; it now asserts hand-computed literals instead.

## [Unreleased] — 2026-08-16 — delay productization line A: shared signal-grid resolver

Plan §2.4. `(sample_rate, fft_size)` admissibility and the frame/hop/bin
derivation used to live in three places on the C side
(`aec_validate_config`'s inline pair table, `aec_derive_dims`'s own
`hop = fft/2`, `AEC3B_RATE_TABLE`'s baked columns) and three more inside
Python's `AecConfig.__post_init__` (a default-fft dict, a `hop = frame // 2`
line, a `valid_grids` dict). Nothing forced them to agree. There is now ONE
table and ONE resolver per language, mirrored row for row. **Output contract
intact**: the same 9 C + 3 Python byte-exact cases as step 1 remain
sample-exact vs `5cd14a0`.

### Added

1. **`aec_resolve_signal_grid()` + `AecSignalGrid`** (C) /
   **`resolve_signal_grid()` + `SignalGrid`** (Python) — the single source
   of truth for `{frame_size, hop_size, n_freqs, is_legacy}`.
   `aec_validate_config()`, `aec_derive_dims()` (hence `aec_get_mem_size()`
   / `aec_create()` / `aec_init()`) and `aec_is_valid_sample_rate()` are all
   thin readers of it; `AecConfig.__post_init__` is the Python resolver's
   sole caller.
2. **`aec_default_fft_size()`** (C) / **`default_fft_size()`** (Python) —
   the convenience-default policy in ONE place, behind
   `aec_config_defaults()` and behind Python's `frame_size = -1` sentinel.
   Exposed so a CLI can resolve an unspecified grid itself instead of
   hard-coding its own copy of the mapping.
3. **Tests** — `python/tests/test_signal_grid.py` (16 cases; the Python
   table is checked against C's `AEC_GRID_TABLE` **parsed out of
   `c_impl/src/aec.c`**, so an unmirrored edit on either side fails) and
   `test_config_validation.c`'s `test_signal_grid_resolver()` (+103 cases:
   resolver content on all four grids, the get-mem-size == init == process-hop
   lockstep on both construction paths, `AEC3B_RATE_TABLE` pinned against the
   resolver, `fft_size = 0` and every mismatched pair rejected by all three
   entry points).

### Changed

4. **The core refuses `fft_size = 0` ("auto")** — it always did in effect
   (no pair matched), but it is now the explicit, documented contract rather
   than a side effect: 16 kHz has two production grids and a library that
   guessed would silently pick one. `aec_config_defaults()` /
   `aec_config_from_preset()` remain the only place a grid is chosen for a
   caller, and `example/aec_wav.c` now resolves the pair itself before
   calling in (clearer error, before any allocation).
5. **8 kHz is flagged `is_legacy`** (plan §11.2, decided: keep as a legacy
   grid, do not deprecate). Supported by this library and the Audio_ALG MONO
   pipeline — real tests depend on it — but not a product grid and not
   supported by the 4-channel pipeline. It is now labelled in the table and
   in the manual instead of sitting anonymously in a guessed path.
6. **`aec_is_valid_sample_rate()` now reads the grid table** rather than a
   separate `AEC_SR_WHITELIST`. Same answers (8000/16000/48000), one fewer
   list to keep in sync. Its contract is documented more sharply: a true
   return does NOT mean any `fft_size` will do — the pair still has to
   resolve.
7. **Python `AEC()` asserts its config is already resolved** — a sentinel or
   a hand-poked inconsistent `frame_size`/`hop_size` raises at construction
   instead of surfacing later as a wrong FFT size deep in the filter.
   Mirrors the C side, where every entry point re-runs the validator.

### Notes

- **Known gap, measured and recorded rather than papered over**: removing
  the resolver call from `aec_validate_config()` alone does NOT turn the new
  test red — rejection still happens one layer later, because
  `aec3b_rate_cfg()` returns NULL for an unsupported pair and
  `aec_get_mem_size()` propagates that as 0. `AEC3B_RATE_TABLE` is therefore
  a de facto second admissibility gate today. The observable public contract
  (all three entry points reject) is pinned either way, and the new
  "`AEC3B_RATE_TABLE` agrees with the resolver" assertion keeps the two
  honest; collapsing the second gate means sourcing that table's four grid
  columns from `AEC_GRID_TABLE`, which is a follow-up, not part of this step.
- Plan §11.3 decided alongside: **NR's grid explicitness is a separate work
  item**; this round does not touch the NR repo.

## [Unreleased] — 2026-08-16 — delay productization line A, step 1: three-state delay mode + validation

First of the four AEC steps in
`docs/delay_estimator_productization_plan_zh_TW.md` §9. Alignment is now
selected by ONE explicit field instead of two implicitly-coupled ones, and
every illegal mode/field combination is rejected rather than normalised.
**Output contract intact**: the default path (`MATCHED`, n=5) and both
legacy spellings render **sample-exact** to the stable baseline `5cd14a0` —
9 C cases (16k/256, 16k/512, 48k/1024 × default/CNG/mild/aggressive/
`--no-delay-est`) and 3 Python cases (default, `from_preset('balanced')`,
legacy `enable_delay_est=False`), compared sample-by-sample rather than by
file hash (float WAV `PEAK` chunks carry a timestamp).

### Added

1. **`AecDelayMode` — `MATCHED` / `FIXED` / `EXTERNAL_ALIGNED`** (C
   `aec.h`, Python `modules/enums.py`, identical integer values 0/1/2).
   `AecConfig` gains `delay_mode` (both ports) and `fixed_delay_samples`
   (C; Python already had it). init-time immutable. `MATCHED` estimates the
   delay online; `FIXED` applies a bring-up-MEASURED delay through the same
   reference ring with no estimator; `EXTERNAL_ALIGNED` builds neither
   estimator nor ring because the caller guarantees `ref` is already
   aligned. Defaults are unchanged (`MATCHED` + n=5 + `fixed=-1`), and the
   C `aec_config_defaults()` spells `fixed_delay_samples = -1` out rather
   than leaving the memset 0 — 0 is a legal fixed delay, so a memset default
   would have made the shipped config itself illegal.
   *All three behaviours already existed; this step NAMES them. The pool is
   deliberately NOT yet differentiated per mode — `FIXED` still carves the
   same ring `MATCHED` does and `DelayAec3` still embeds its compile-time-max
   arrays in every mode. Shrinking `sizeof(Aec)` per mode/n is plan step 2/3.*

2. **`aec_config_resolve_delay()` (C) / `AecConfig._resolve_delay_mode()`
   (Python)** — the one, explicit, idempotent translation layer for the now
   **deprecated** `enable_delay_est` mirror, run before validation on every
   entry point (`aec_get_mem_size` / `aec_create` / `aec_init`;
   `AecConfig.__post_init__` plus a re-run at `AEC()` construction so a
   late `cfg.enable_delay_est = False` poke is still honoured, matching where
   C resolves). Mapping: `est=1` (its default) carries no information and
   leaves `delay_mode` alone; `est=0` + `fixed>=0` → `FIXED`; `est=0` +
   `fixed<0` → `EXTERNAL_ALIGNED`. The mirror is rewritten from the resolved
   mode afterwards, so **only `delay_mode` is the source of truth** from
   that point on. `enable_delay_est` is scheduled for removal once callers
   have moved.

3. **`AecLinearContext` per-mode semantics** (plan appendix §11.1) — the NN
   seam now tells the truth about each mode instead of collapsing two of
   them onto "no estimator": `FIXED` reports `delay_samples ==
   fixed_delay_samples` with `confidence 1.0` and a frozen `generation`;
   `EXTERNAL_ALIGNED` reports `delay_samples == 0`, `LOCKED`, `confidence
   1.0`, frozen `generation`, and its `aligned_far_hop` IS the caller's own
   hop byte for byte. One deliberate refinement of §11.1: `FIXED` stays
   `UNLOCKED` for the first `ceil(fixed/hop)+1` hops, the window where the
   ring cannot yet serve the offset and `far_hop` is still RAW — claiming
   `LOCKED` there would break this seam's central promise ("`UNLOCKED` means
   the content is the RAW far") exactly where a small-search-range consumer
   is most likely to mis-handle it. Steady state is `LOCKED` as specified.

4. **Tests** — `python/tests/test_delay_mode.py` (22 cases: mapping table,
   int/string coercion, `replace()`/`asdict()` idempotency, legacy-vs-explicit
   bit-identical audio for both non-MATCHED modes, the illegal-combination
   matrix, per-mode wiring, late-mutation translation);
   `test_config_validation.c` +106 cases (same mapping table row for row, all
   three entry points on every illegal combination, plus a "guard is not
   blanket" sweep proving each legal combination still constructs on both the
   heap and caller-pool paths); `test_linear_context.c` +55 cases (per-mode
   seam assertions incl. both legacy spellings, and a legal `FIXED 0`).
   Mutation-checked: dropping the `fixed>=0` arm of the translation, dropping
   the `enable_delay_est` guard, or moving the ring write/read back under the
   estimator gate each turns them red (Python 3/6, C 3/4 failures).

### Changed

5. **`fixed_delay_samples >= 0` no longer silently overrides
   `enable_delay_est`** (Python; the C side never had the field). It was an
   undocumented implicit mode switch inside the orchestrator:
   `AecConfig(fixed_delay_samples=1600)` ran in fixed-delay mode while
   `enable_delay_est` still read `True`. It now **raises** — say
   `delay_mode=AecDelayMode.FIXED` (or, legacy-style, `enable_delay_est=False`)
   alongside it. This is the only source-compatibility break in this step;
   no in-repo caller used that shape.

6. **`delay_num_filters` outside `MATCHED` is rejected**, not ignored (both
   ports). There is no matched filter to size in `FIXED` /
   `EXTERNAL_ALIGNED`, and accepting `n=2` there would let a caller believe
   they had bought a compute saving those modes already give them in full.
   `delay_num_filters=0` remains an error in every mode — never a silent
   "delay estimation off" switch.

7. **`AecLinearContext` for `enable_delay_est=0` reads `delay_samples=0` /
   `LOCKED`** where it previously read `−1` / `UNLOCKED` (see item 3). A
   seam-contract change, not an audio change; downstream consumers
   (`Audio_ALG` pipelines, AIAEC) are updated in the pipeline steps of the
   plan. `AecDebugStatus.delay_confidence` likewise reports `1.0` for
   `FIXED` instead of reading the never-initialised `DelayAec3`, and `1.0`
   for `EXTERNAL_ALIGNED` too (it read `0.0` before): both modes take their
   alignment from a caller contract rather than from an estimator, so both
   carry the same "no uncertainty to report" value that
   `AecLinearContext.delay_confidence` has always reported for them. A
   DIAGNOSTIC-field change only — no audio path reads it.

8. **`Aec.has_delay` now means "the matched-filter ESTIMATOR exists"**
   (i.e. `MATCHED` only), matching Python's `delay_est is not None`. "Does a
   reference ring exist" is `cfg.delay_mode != AEC_DELAY_EXTERNAL_ALIGNED`
   (Python's `_delay_active`). The AEC3 post chain's `delay_active` input was
   reading the former where it needed the latter — inconsequential until now
   (the two agreed in both C-reachable modes), but wrong for `FIXED`, which
   would have reported no external delay while applying one every hop.

### Notes

- `AecConfig` grew two fields, appended at the END of the struct so every
  pre-existing field keeps its byte offset. `sizeof(AecConfig)` and hence
  `sizeof(Aec)` / `aec_get_mem_size()` change: **rebuild required**, no
  source change required.
- The C `FIXED` path is new implementation (the field did not exist in C
  before), mirroring the Python orchestrator's structure exactly: the OUTER
  gate is "is there a ring", the INNER gate is "is there an estimator". The
  ring grows to `fixed_delay_samples + 4096` when that exceeds the
  `max_delay_ms` budget, again mirroring Python.

## [Unreleased] — 2026-08-14 — delay-estimator round: configurable bank size, delay survey docs (pre-echo fix held on branch)

Split landing (2026-08-15): the behaviour-neutral items below are ON main;
the pre-echo winner fix is NOT — it lives on `feature/delay-prescan-anchor`
pending the isolated 800-case blind A/B (it perturbs 25% of the 2021 blind
set with a statistically unresolved AECMOS balance, see item 1).

### Pending on branch (NOT in main)

1. **Pre-echo instantaneous error from the WINNING filter**
   (Python `700994b`, C `524a234`). Both ports previously fed
   the shared prefix-error buffer from the LAST filter and applied it to
   `accumulated_error[winner_index]` — correct only when the winner happened
   to be the last filter, which neutered pre-echo onset detection (walk-back
   always broke on the ~1.0 unexcited-region error). Upstream computes it
   for `n == last_detected_best_lag_filter`; both ports now match. NOT
   latent at corpus level (an earlier single-clip observation suggested
   otherwise; the engagement counter is CUMULATIVE, not consecutive):
   blind-set A/B (2021 full 287 pairs, isolated to exactly these two
   commits) shows **72/287 clips (25%) change via altered delay
   trajectories**, concentrated on movement material, both directions
   (stability gains and oscillation regressions). AECMOS on the differing
   clips is a statistical wash (bucket means within ±0.05, per-clip swings
   to 1.46, NE untouched 0/33) — below the resolution of 287 clips, hence
   the isolated 800-case gate before merge. Synthetic two-arrival cases
   resolve the onset correctly (fixed `(lag,pre_echo)=(60,27)` vs pre-fix
   `(60,60)`).

### Added

2. **`delay_num_filters` config (Python + C)** — matched-filter bank size
   1..5, default 5 = byte-equal geometry (tests pin ring capacity 2064 /
   histogram 2433 and bit-identical output vs default). Compute knob for
   deployments whose bulk system delay is already compensated via
   `fixed_delay_samples`: reliable reach 125/221/317/413/509 ms for n=1..5,
   ~4.2 MMAC/s full-rate saved per dropped filter; C arrays stay at the
   compile-time bound (RAM unchanged, pool contract intact). Geometry
   proven end-to-end: n=2 locks a 150 ms echo, refuses a 350 ms echo that
   n=5 locks (`test_delay_num_filters.{py,c}`, wiring-mutation checked).

3. **`docs/delay_estimator_design_zh_TW.md`** — matched-filter mechanism,
   confidence semantics (histogram consistency, not match quality; the
   out-of-range confident-mislock mode), num_filters provenance, blind-set
   delay survey (2021 full: p50 48 ms / p90 225 ms / max 923 ms, ~5% beyond
   509 ms; 2023 sample 48 kHz: all ≤349 ms), 48 kHz applicability audit,
   and the system-layer delay stance. Survey tool: `wav/measure_blind_delay.py`.

## [Unreleased] — 2026-08-13 — AecLinearContext seam (aligned far + delay status + generation token), ref_ring_filled overflow fix

New public surface for an external neural post-filter (Align-ULCNet-class
RES+NR) that needs the time-domain aligned far-end plus delay status. Output
contract intact: `aec_wav` renders verified **byte-equal** to 4.0.0 on the
challenge doubletalk clip at 16k/256+CNG and 16k/512 (full default path,
delay est active).

### Added

1. **`AecLinearContext` + `aec_get_linear_context()`** (aec.h): read-only,
   zero-copy view of `{formed_linear_hop, aligned_far_hop, delay_samples,
   delay_confidence, delay_state, generation}`. `aligned_far_hop` aliases the
   internal `far_hop` — byte-identical to what the PBFDKF consumed this hop.
   `delay_state` is honest about non-alignment: `UNLOCKED` covers
   pre-acquisition, the unfilled-ring window, and `enable_delay_est=0`
   (where `delay_samples` reads −1, not the internal 0). `generation` is a
   saturating token bumped at every ring-offset change — first acquisition,
   confirmed shift (the soft-recovery realign paths set no other flag, so
   this is their only external trace), and `aec_reset()`. `CHANGED` is
   reported exactly on the hop that bumped it. Doc:
   `docs/nn_integration_interface.md` (new time-domain seam section).
2. **`test/test_linear_context.c`** (`make test-linear-context`): alias +
   byte-equal content proof against the caller's own delayed raw far, on
   16k/256, 16k/512, 48k/1024; generation/CHANGED at acquisition, shift and
   reset; honest-UNLOCKED window (state-poked); delay-est-off honesty; ring
   fill saturation. Mutation-tested: dropping either generation bump, forcing
   `far_hop_aligned=1`, or removing the fill clamp each turns it red
   (6/1/2 failures respectively). Measured info: post-lock delay-shift
   re-lock takes 16.3 s of audio at 16k/256, 7.4 s at 16k/512, 35 s at 48k in
   the `enable_res=0` context-only config — where the ERLE watchdog is inert
   (`last_erle_windowed` is only cached under `enable_res`, aec.c) — so
   integrators should not expect fast re-lock from this seam alone.

### Fixed

3. **`ref_ring_filled` signed-overflow UB** (aec.c ring-write step): the
   monotone fill counter now freezes at `ref_ring_size` (guard BEFORE the
   increment, freeze-not-wrap like the other counters). Unbounded, it
   overflowed after ~37 h of continuous 16 kHz audio. Covered in
   `test_counter_saturation.c` (INT_MAX-seeded, UBSan wrapper aborts on the
   reverted form) — this counter was the one miss of the earlier
   counter-saturation sweep.
4. **Context-only ERLE cache port divergence** (was listed under Notes as
   known-not-fixed; fixed in a follow-up commit): `last_erle_windowed` is now
   cached under `enable_res || return_res_context`, matching both the compute
   block right above it and the Python spec. In the context-only seam config
   the already-cancelling Path-A guard and the duty ERLE watchdog come back
   to life (Python always had them); measured effect on the seam test's
   delay-shift re-lock: 16.3 s -> 13.4 s of audio at 16k/256, 7.4 s -> 4.7 s
   at 16k/512, 48 k unchanged. Default paths (`enable_res=1`) byte-equal
   (re-verified with an `aec_wav` render diff). The warm tap-transfer gate
   was never affected (inst-ERLE ring fills unconditionally).
5. **Ring-capacity defense at acquisition**: a candidate delay larger than
   `ref_ring_size - hop` is no longer eligible — previously the modulo read
   would alias and silently return wrong (effectively future) far. Unreachable
   with default configs (2048 ms ring vs ~509 ms search span); reachable only
   when a caller shrinks `delay_buffer_ms`/`max_delay_ms`.

### Notes

- Pool sizes grow by 16 B per instance (three bookkeeping fields):
  16k/256 381 056 → 381 072, 16k/512 510 128 → 510 144,
  48k/1024 1 166 976 → 1 166 992.
- `python/aec.py` and everything under `python/` is intentionally untouched
  (no `__version__` bump yet): the AIAEC dataset contract pins an AST hash of
  the mirrored `lib/aec/python` tree, and any Python-side edit — including a
  version-string bump — invalidates every existing packed shard and
  checkpoint. Fold the version bump into the next coordinated
  contract-breaking release.

## [4.0.0] — 2026-08-06 — Output-contract + public-ABI break: custom output limiter removed, 16 kHz default grid 512/256 → 256/128, context-only entry points, one shared FftHandle

The `4.x` series is the same v3.21/v3.22 AEC3 algorithm; the major bump
records the ABI and output-contract breaks below, not a new algorithm
generation. The production chain is still the v3.21 AEC3-aligned `_aec3_post`
chain with the v3.22 split min-gain floor.

### ⚠ BREAKING

1. **Output is no longer bit-identical.** The custom peak-ratio time-domain
   output limiter was removed from the last stage of `aec_process()`. The
   `aec_wav` + `cmp` byte-equality gate does not apply across this release.
   *Migration*: re-baseline every stored reference render and every
   byte-equality gate; if you relied on the limiter for clipping protection,
   add an AGC/DRC or output-stage limiter outside this library. Measured on
   75 AEC Challenge blind clips (25 each far-end singletalk / doubletalk /
   near-end singletalk, KISS backend, BALANCED): mean level change +0.05 to
   +0.14 dB (max +0.59 dB); files containing any sample above 1.0 went from
   16/75 to 17/75; worst-case peak 1.156 → 1.381, affecting ≤0.09 % of samples
   in the worst file, every such file having a microphone peak ≥0.96 (source
   already at full scale). No sustained excursion or divergence was observed.
   Note the old limiter did **not** prevent overshoot either — it reached
   1.156/1.149/1.116 on those same files, because its gain is smoothed and
   one-hop-lagged and so cannot catch an intra-hop transient.
   - *Relaxation, not a break, but it belongs with this item*: **the
     full/context entry-point mixing restriction is lifted.** The limiter was
     the only state that `aec_process()` advanced and `aec_process_context()`
     did not, so the "never mix classes within one construct-or-reset epoch"
     rule previously documented on `aec_process_context()` is gone.
     `aec_process()`, `aec_process_capture()`, `aec_process_context()` and
     `aec_process_context_shared_far()` may now be interleaved freely on one
     instance. *Migration*: any integration guide that copied the old
     hard-precondition rule must be corrected.
2. **`--debug-trace` CSV schema change.** The `limiter_gain` column is gone; a
   row is now 14 fields, ending at `raw_err_pwr` (was 15). *Migration*: update
   every downstream trace parser — column indices after the removed field all
   shift.
3. **Public `AecConfig` struct-layout change (twice).**
   `spatial_linear_context` was added after `return_res_context`, and
   `return_formed_output` was added. *Migration*: C callers must rebuild
   against the updated header — do not link an old build's object files or a
   cached static library against the new header (or vice versa); a mismatched
   `sizeof(AecConfig)`/field offsets between a caller's compiled code and this
   library is undefined behavior, not merely a stale-default risk. A
   zero-initialized (`memset`) `AecConfig` from a *rebuilt* caller is
   unaffected — both new fields default to 0 = off, matching every existing
   caller's behavior.
4. **`AecResContext` changed additively AND semantically.** New `formed_hop`
   field (C) / `formed_output` (Python), so C callers must rebuild. More
   importantly `error_spec` now *always* exposes the selected linear output on
   the post-filter's periodic-sqrt-Hann, 50%-overlap WOLA grid; the no-shadow
   C/Python path previously exposed PBFDKF's internal estimator spectrum,
   which is not a reconstructing STFT of the continuous linear output — so
   no-shadow consumers get different NUMBERS, not just a new field.
   `echo_spec` and `near_spec` are re-aligned to the same window and frame
   alignment, with `near_spec = error_spec + echo_spec`. Additionally
   `res_gain` is `NULL` (not a stale/zeroed array) whenever
   `spatial_linear_context` is set. *Migration*: rebuild; re-derive any
   downstream RES/NR tuning calibrated against the old no-shadow `error_spec`;
   NULL-check `res_gain` when using `spatial_linear_context`.
5. **16 kHz default signal grid 512/256 → 256/128 (8 ms hop).** For every
   caller that leaves `frame_size`/`fft_size` unspecified this changes default
   algorithmic latency 16 ms → 8 ms, default static pool 543,040 B → 397,072 B,
   and `n_partitions` 4 → 7 (`ceil(filter_length/hop)` with
   `filter_length = 832`). *Migration*: callers that need the old grid must now
   request it explicitly (`AecConfig(sample_rate=16000, frame_size=...)` /
   `cfg->fft_size`); both 16 kHz grids remain fully supported. 8 kHz (256) and
   48 kHz (1024) defaults are unchanged.
6. **One shared `FftHandle` per `Aec` instance instead of three.**
   `pbfdaf_init_static()` gained a REQUIRED `shared_fft` parameter, and
   `pbfdaf_free()`'s static path no longer calls `fft_destroy` (it never owns
   the borrowed handle; calling it would double-free the NE10 backend's shared
   twiddle-config allocation). *Migration*: update out-of-repo
   `pbfdaf_init_static()` call sites; re-read `aec_get_mem_size()` instead of
   using any previously published pool figure — every one of them changed:
   16k/256 −17,568 B KISS / −16,352 B NE10; 16k/512 −33,952 B KISS /
   −31,200 B NE10; 48k/1024 −66,720 B KISS / −60,896 B NE10.
7. **`epc_init()` signature change (C).**
   `epc_init(EpcDetector*, int hangover, float total_rise, float
   delta_threshold)` gained two trailing parameters, `int hop_size, int
   sample_rate`, and `EpcDetector` gained two new fields, `float epv_fast_tc,
   epv_slow_tc`. Python's `EchoPathChangeDetector.__init__` gained the same
   two as optional keyword-only arguments. *Migration*: all in-repo call sites
   are updated; out-of-repo callers must pass the real grid.
8. **Production 16 kHz behaviour change — AEC3 constant audit, Tier 2.**
   `subband_erle` (`alpha_up`/`alpha_down`/`onset_release_decay`),
   `fullband_erle` (`quality_alpha`/`td_alpha`/`maxmin_forget`),
   `residual_echo_estimator` (`noise_floor_growth_per_hop`, 1.1 → ~1.27 at
   hop=160) and `stationarity_estimator` (noise-spectrum `alpha`/`alpha_init`)
   were previously applied as raw, unrescaled AEC3 literals with no rate
   conversion at all, so rescaling them for the live `sample_rate` changes
   today's 16 kHz output. These four have **not** been through the 800-case
   AECMOS bench yet — do not treat as ship-ready. *Migration*: re-validate
   any tuning derived from the previous 16 kHz output.
9. **Production 16 kHz behaviour change — top-level hop-authored constants
   retimed to wall clock.** `shadow_err_alpha`, `warmup_frames`,
   `epc_hangover`, `ne_recent_hold`, `filter_misadjustment_stable_frames`,
   `filter_misadjustment_hangover_frames` and
   `EchoPathChangeDetector.EPV_FAST_TC`/`EPV_SLOW_TC` were literal hop counts
   / per-hop EMA constants frozen at the legacy hop=160/sample_rate=16000
   (10 ms) grid. Measured end-to-end drift at 16 kHz: linear-path max diff
   1.3e-5, post-chain max diff 1.9e-4 (both well inside the 2.0e-2 gate).
   *Migration*: same as #8 — a genuine behavior change, not byte-equal.
10. **8 kHz timing behaviour change.** `delay_aec3_init()` was hardcoding the
    internal delay estimator's feed rate to 16000 regardless of the AEC's
    configured `sample_rate`, so at native 8 kHz both real-time thresholds (the
    ~30 s clockdrift `stability_reset_hops` and the ~500 ms
    `consistent_estimate_threshold`) ran at 2x their intended wall-clock
    duration. Fixed to pass the true feed rate. 16 kHz was coincidentally
    correct. *Migration*: re-validate any 8 kHz delay-acquisition timing
    expectation.
11. **Downstream contract bumps.** `LINEAR_AEC_CONTRACT_VERSION` v1 → v2 —
    AI-AEC `dataset_gen`'s ch5 channel read the limiter-processed output, so
    *migration*: any dataset sequence built against v1 must be regenerated with
    `rematerialize_linear_aec.py`. `FOUR_AEC_NR_RES_LAYOUT_VERSION` also
    bumped, because the 4ch wrapper's `lane_out` staging buffer was removed.
12. **Config validation tightened.** `highpass_cutoff_hz` now mirrors
    audio_common's `hpf_params_valid` exactly (isfinite / >0 / <0.45·sr) when
    `enable_highpass` is set. Configs such as {8 kHz, 4 kHz} used to pass the
    flat [0, 20000] bound and then silently construct with NO mic-path HPF
    (`hpf_init` NULL was never checked); `aec_carve` now FAILS CONSTRUCTION on
    them. *Migration*: fix out-of-band cutoffs — they now hard-fail instead of
    silently disabling the filter.
13. **Build-invocation break.** `make CFLAGS=...` (likewise `CXXFLAGS`,
    `CPPFLAGS`, `LDFLAGS`, `FP_POLICY`) is now rejected at parse time, because
    GNU make silently drops the Makefile's own `+=`/`:=` appends for a
    command-line-set variable — which used to strip `-ffp-contract=off` and
    `-DAEC_NO_STDIO` while still building. *Migration*: any build script
    passing those variables now hard-errors; move to `EXTRA_CFLAGS` /
    `EXTRA_LDFLAGS`, the only supported hooks.
14. **Release-flow break.** `make publish` now FATALs by default when this
    checkout OR the resolved audio_common producer checkout has uncommitted
    tracked changes, OR contains any untracked non-ignored file, OR has no git
    identity at all (the last is refused unconditionally — no escape hatch).
    `ALLOW_DIRTY_PUBLISH=1` and `ALLOW_UNTRACKED_PUBLISH=1` are separate,
    orthogonal knobs. `make -t publish` is now an explicit no-op.
    *Migration*: commit or ignore working-tree content before publishing.
    NOTE: this repo currently carries ~98 dirty entries including ~75
    untracked moved files, so `publish` WILL refuse until they are committed.
15. **Preset renamed `gentle` → `mild`** (`AEC_PRESET_MILD`; NR-style naming,
    parameters unchanged). *Migration*: update preset strings/enums at call
    sites.

16. **Detector timing constants are now retimed per grid, and three internal
    init signatures changed.** `render_activity_init`, `filter_convergence_init`
    and `doubletalk_init` each gained trailing `int hop_size, int sample_rate`
    (`detectors.h`); the coefficients they used to read from file-scope statics
    now live in the `RenderActivity` / `FilterConvergence` / `DoubleTalk`
    structs. Both struct layouts changed. These are internal headers, not part
    of the `aec.h` public surface, but anything that constructs the detectors
    directly must be updated. *Migration*: pass the instance's hop and sample
    rate at init. Note `filter_convergence_reset()` no longer forwards to
    `filter_convergence_init()`, because reset has no grid to re-derive from.

17. **Five adaptation constants are now retimed per grid, and
    `saturation_init()` changed signature.** The 2026-08-06 detector pass (16)
    covered the constants reachable from `AecConfig`; these five live in the
    adaptation path and were missed, so they still covered a different
    wall-clock span at every grid:

    | constant | where | authoring grid | wall-clock TC (all grids, after) |
    |---|---|---|---|
    | `Aec::alpha_pow` | `aec.c` | per-**sample**, sr=16000 | 1.2185 ms |
    | `Aec::alpha_erl_tracking` | `aec.c` | per-hop, 10 ms | 994.99 ms |
    | `Aec::alpha_erl_converged` | `aec.c` | per-hop, 10 ms | 9995.00 ms |
    | `PBFDAF::alpha_power` | `pbfdkf.c` | per-hop, 10 ms | 94.91 ms |
    | `PBFDKF::alpha_r` | `pbfdkf.c` | per-hop, 10 ms | 194.96 ms |
    | `Saturation::alpha_attack/release` | `saturation.c` | per-hop, **16 ms** | 13.29 / 791.97 ms |

    The reference grids are **not uniform** — verified per constant from git
    provenance (`243d67c` authored the saturation pair against frame 512 /
    hop 256). Retiming a 16 ms constant off the 10 ms reference is wrong by
    exactly 1.6x, and is invisible unless a 10 ms-hop grid is sampled.
    Consequently the saturation pair is **unchanged** at 8000/256 and
    16000/512 — both already have a 16.000 ms hop. That is correct behaviour,
    not an unfinished retime.

    `alpha_r` shipped in this release on the **10 ms** anchor, but reached it
    in two steps and the intermediate state is worth recording. It was first
    retimed off its *introducing* commit (`e9cb383`, 2026-03-20, frame 512 /
    hop 256 → 311.93 ms). That is the wrong criterion: the anchor is the span
    that was last empirically *validated*, and the default grid moved to
    320/160 three days later at `9735b0f` — which re-benched at the new grid —
    after which the EMA was re-authored at 10 ms by `c2551db` (the v3.0.0
    filter rewrite, fixes B-11 and B-3a targeting this exact EMA) and 10 ms
    held through every campaign until `d862a38`. `d7e94f7` corrected the
    reference hop from 256 to 160 in both ports, giving 194.96 ms. A reader
    comparing against a pre-`d7e94f7` document will see 311.93 ms; 194.96 ms
    is current.

    `saturation_init()` gained trailing `int hop_size, int sample_rate`
    (`saturation.h`). *Migration*: pass the instance's hop and sample rate.
    Behaviour is unchanged at 8 kHz/256 and 16 kHz/512 and changes elsewhere.

    Validated by a 90-case blind A/B (`preset=balanced`, `--filter 52`,
    `--cng`, `NO_PREALIGN=1`) on both 16 kHz grids: worst bucket ΔAECMOS-echo
    −0.017, worst Δdeg −0.013, every bucket within the ±0.05 abort band. 48 kHz
    has a structural pass only — see "Known limitations". Per-case scores, the
    exact baseline→candidate diff and the harness are checked in under
    `eval/ab_evidence/2026-08-06-adaptation-retiming/`.

18. **`pbfdaf_init()` / `pbfdkf_init()` and their `_static()` counterparts
    now reject a non-positive `sample_rate`, resolved hop, or NULL instance**
    with `-1`, writing nothing to either the instance or caller-owned pool.
    These are public API sitting under `aec_create()`'s validator, and since
    (17) `sample_rate` is load-bearing there. *Migration*: direct callers must
    pass a real sample rate; `0` no longer means "don't care". `saturation_init()`
    returns void and cannot report the error, so `aec3_growth_rehop()` absorbs
    it by returning the authoring value — leaving the detector un-retimed
    rather than driving it with a NaN retention.

19. **The F2.4 simple-mu retime was evaluated and rejected; production keeps
    the four hop-authored values together.** The candidate converted the
    holdoff and attack/hold/release alphas to a common wall-clock reference,
    but the mechanism is a branch-coupled state machine rather than four
    independent timers. On the two-grid 90-case A/B it produced case-level
    Δecho down to −0.257 and Δdeg down to −0.349; the 2⁴ interaction matrix was
    strongly non-additive and did not transfer between grids.

    Production therefore remains at `20 / 0.3 / 0.99 / 0.95`. There is no
    `Aec` layout or pool-size change and no migration requirement. The known
    grid-dependent timing is recorded as a rejected-retime exception in
    `docs/timing_constant_inventory.md`; complete evidence is under
    `eval/ab_evidence/2026-08-07-simple-mu/`.

### Added

- **`test_rate_structural` check (d6)** and
  `python/tests/test_simple_mu_holdoff_f24.py` lock the rejected-retime
  exception on all four grids. They recover the coefficient each branch
  actually applies and test the arm site independently, while the F2.4
  no-reset guard remains covered.

- **`test_rate_structural` check (d5)** + `python/tests/test_alpha_r_direct_pbfdkf.py`
  — the gate for `alpha_r`, which no wrapper-level benchmark can observe: it is
  inert through every `Aec` path (for two *different* reasons depending on
  shadow mode) and live only for a direct `pbfdkf_process()` caller. Both drive
  PBFDKF directly and recover the applied coefficient. See
  `eval/ab_evidence/2026-08-07-alpha-r/`.

- **`eval/ab_compare.py`** + `python/tests/test_ab_compare.py` — the
  completeness gate for a blind A/B. The rendered, scored and manifest stem
  *sets* must be equal (not the same size), every output pair is compared by
  SHA-256 and per-sample statistics into `wav_comparison.json`, and a NaN in
  the candidate aborts rather than reading as identical to a tolerance check.
  Every guard is mutation-tested. `eval/run_c_ab.sh` is the one canonical
  harness; it forwards extra arguments to `aec_wav`, builds with `make clean`
  and `WERROR=1`, and aborts if the two builds are byte-identical.

- **`docs/timing_constant_inventory.json`** — the inventory's data is now
  repository state, and `python/diag/gen_timing_inventory.py --check` rebuilds
  the Markdown in memory and fails on divergence. The previous generator read
  an absolute path under a developer's home directory.

- **`test_rate_structural` check (d2)** + `python/tests/test_detector_timing_effective_values.py`
  — the permanent regression gate for (17), on all four grids in both
  languages. Both assert the value an instance **actually ends up with**, not
  that a retiming call appears in the source: a reviewer reading the four
  touched modules concluded the retiming had never been applied, because three
  of the seven values are legitimately identical to their authored literal on
  two of the four grids. Only an effective-value assertion separates "retimed"
  from "not retimed". Both halves of the negative space are covered — the 16 ms
  saturation pair must NOT move on a 16 ms grid, and the 10 ms/per-sample
  constants MUST move where the grids differ, `alpha_r` included since its
  anchor moved. Mutation-tested: reverting any one constant to its authored
  literal fails it, and so does retiming a 16 ms constant off the 10 ms
  reference (`alpha_release` would read 494.98 ms against the expected
  791.97 ms).

- **`make test-config-validation`** — `test_config_validation.c` had lived
  since 2026-07 as a standalone-gcc recipe with **no Makefile target**, so its
  checks only ran when someone remembered to compile it by hand. Now wired, and
  extended with `test_direct_init_rejects_bad_rate` covering (18).

- **`make test-detectors-parity`** + `python/diag/gen_detectors_golden.py` —
  the Python/C gate for `detectors.{c,h}`. `test/modules/parity_detectors.c`
  had existed since 2026-07 with **no Makefile target and no golden
  generator**, so it had never run once. It also asserted fp64 bit-exactness on
  three float fields, which `detectors.c` retired long ago, so it could not
  have passed even if invoked — which is presumably why it was shelved rather
  than fixed. Both gaps are closed: booleans and counters compare exactly,
  instantaneous floats in an fp32 ULP band, EMA-accumulated floats in a
  relative band (a fixed ULP count cannot express drift that grows with the
  length of a decay run once the two ports' EMA coefficients differ).
  The golden regenerates on every invocation, so the gate compares the CURRENT
  Python against the CURRENT C rather than a stale blob, and the header carries
  `sample_rate` as well as `hop` (hop=128 alone cannot distinguish 8 kHz from
  16 kHz, and they retime differently). Mutation-tested: reverting the shadow
  gate, dropping any individual retime, using the wrong reference grid, or
  breaking the `NEW = 1 - OLD` pairing each fail it.

- **`aec_process_context(aec, mic, ref)`** — context-only entry point.
  `aec_process`'s ~900-line body is now split into a shared static
  `aec_process_core()` (the linear filter + AEC3 post/RES block, plus the
  power-EMA/convergence-detection steps the NEXT hop depends on) and two thin
  wrappers: `aec_process` (core + emit into `out`) and `aec_process_context`
  (core only — no `out` parameter). For a caller that only reads
  `aec_get_res_context()` (`error_spec`/`res_gain`/`formed_hop`/etc, e.g. a
  pipeline running `enable_res=0 && return_res_context=1` purely for the
  linear filter's context) and never plays `aec_process`'s own returned audio,
  this skips the final `O(hop)` emit copy — the linear filter and RES/post
  cost are identical either way. Wired into the Makefile as
  `make test-process-context` (`test/test_process_context.c`, 74 checks:
  byte-identical `AecResContext` — every field a real caller reads, including
  `linear_hop`/`far_spec`/`res_gain`/`saturation_level` — between
  `aec_process` and `aec_process_context` across the 3 officially-contracted
  grids x shadow on/off x `spatial_linear_context` on/off, i.e. the 4ch
  wrapper's exact per-lane config; every populated field checked finite; a
  reset-survival case; and, since the limiter's removal, a case that
  interleaves the full and context-only entry points and requires
  bit-identical output against an `aec_process()`-only reference).
- **`aec_process_context_shared_far(aec, mic, ref, shared_far_spec)`** — like
  `aec_process_context`, but a non-NULL `shared_far_spec` lets this instance
  skip computing its own far-end FFT and borrow one an external caller already
  computed, for a multi-instance caller (e.g. a 4-lane spatial wrapper) whose
  instances all see the identical far-end signal every hop. Generalizes the
  pre-existing internal shadow→main `precomputed_far_spec` borrow mechanism
  (documented as "borrowed, one-shot, cleared after use") to an external
  source; `ref` is still required even when sharing (the OLA far-buffer
  history and every non-FFT use of the raw far signal — saturation detection,
  delay estimation, mu_scale — are unaffected).
- **`aec_far_fft_real_compute_count(aec)`** — read-only instrumentation:
  cumulative count of hops this instance actually ran its own far-end rfft
  (rather than borrowing one), for tests proving a multi-lane caller's total
  far-FFT count really dropped. Covered by `test/test_shared_far_spec.c`
  (51 checks, `make test-shared-far-spec`): byte-identical `AecResContext`
  between borrowing and independently computing across all 3 grids x shadow
  on/off; a negative control proving a wrong shared spectrum genuinely changes
  the result (rules out a silent fallback); a heap-allocate-then-free
  reset-safety case; and the counter's 4-hop mutation test.
- **`AecConfig.return_formed_output`** (default `False`, no behavior change
  unless set). When `True`, `process()`'s return shape is unchanged and
  `aec.get_formed_output()` becomes available after each call, returning the
  same value `AecResContext.formed_output` exposes — the shadow/main-SELECTED,
  crossfaded, WOLA-formed linear residual — without requiring
  `return_res_context=True`'s full `AecResContext` (and, when
  `enable_res=False`, without running the rest of the `_aec3_post` gain/CNG
  chain that context would otherwise pull in just to populate one field). Runs
  only the AEC3 `UseRefinedOutput`/`FormLinearFilterOutput`
  selection-and-crossfade step (`_aec3_select_linear_filter_output()`); does
  not feed back into filter tap adaptation. Verified byte-identical to
  `context.formed_output` across 3 grids x shadow on/off — see
  `python/tests/test_formed_output_seam.py`. The seam is not merely
  "pre-limiter": it exposes the selected/crossfaded linear residual, which is
  not the raw main-filter output, so consumers that need the linear residual
  (AI-AEC frontend) must keep reading it.
- **`AecConfig.spatial_linear_context`** (default 0, no behavior change unless
  set). When enabled, this AEC instance never computes its own
  `SuppressionGain` output (`res_gain`) — intended for a multi-channel caller
  (e.g. `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c`) that runs one AEC per
  microphone lane purely for its linear filter, then recomputes an equivalent
  gain once from beamformed multi-lane data; a per-lane gain in that
  architecture is computed and then never read. The `DominantNearend`
  hold-state (`suppression_gain_update_dominant_nearend()`, newly exported
  from `suppression_gain.h`) still updates every hop even in this mode: it
  feeds the next hop's ERLE onset decision, and therefore `r2`, which remains
  used. `comfort_noise` is computed independently in Step 19 — it is one of
  `DominantNearend`'s own *inputs*, not something it affects — and stays
  correct in this mode regardless. Only the remaining, provably-dead part of
  `suppression_gain_get_gain()` is skipped. Verified bit-exact against the
  normal path on every other `AecResContext` field (`r2`, `comfort_noise`,
  `near_spec`, `echo_spec`, `error_spec`, `formed_hop`) and the synthesized
  output, at every supported grid, shadow filter on and off
  (`c_impl/test/test_rate_structural.c`). Only valid together with the
  existing context-only seam (`enable_res=0 && return_res_context=1`);
  `aec_validate_config()` rejects any other combination.
- **`AecResContext.formed_hop` (C) / `formed_output` (Python)** — the exact
  current selected/crossfaded time-domain hop represented by the second half
  of the WOLA frame, plus structural reconstruction tests for every supported
  grid with the shadow filter both enabled and disabled.
- **`aec3_growth_rehop()` / `growth_rehop()`** (`aec3_scale.c`/`.h`/`.py`) —
  generalizes the existing `aec3_per_block_growth_to_per_hop` (already a
  direct power law, already the right convention) from its hardcoded AEC3 4 ms
  block reference to an arbitrary reference period (our legacy
  hop=160/sample_rate=16000). No new conversion *mechanism* — same power-law
  derivation, parameterized reference.
- **Timing-audit tests**: `c_impl/test/test_rate_structural.c` gained a sixth
  check, `test_top_level_constant_retiming` (40 new assertions across all four
  grids, exercising the post-defaults `cfg.fft_size` override path), plus a
  new `python/tests/test_hop_authored_timing_parity.py` (9 tests). Both
  mutation-tested (reverting the C `aec_carve()` retiming block, the C
  `epc_init()` EPV retiming, and the Python `__post_init__` retiming each
  independently, confirming the new tests fail without the fix and pass with
  it).
- **`make test-zero-heap`** — an allocator-hook release gate for the complete
  caller-pool init/process/reset/destroy lifecycle on KISS and NE10.
- **Multi-rate 8/16/48 kHz (F01)**: every dimension derived from the
  hop = 10 ms rule. Per-rate tables are generator-emitted from the live Python
  spec (`gen_aec_balanced_config_h.py` — its cross-rate invariance assertion
  caught 8 genuinely rate-scaled power constants), consumed via
  `aec3b_rate_cfg()`; `filter_length` is ms-derived (416/832/3072).
  Acceptance: per-rate Python↔C e2e parity (8 k −90 dB corr 1.0; 48 k
  −23.8 dB corr 0.99998, root-caused and tolerance-documented), delay
  C-goldens bit-exact ×3, static==dynamic ×3×2 backends, COLA/impulse
  structural tests, UBSan clean ×3.
- **Test/validation surface from the external-review campaign**:
  `test_zero_heap_aec.c`, `test_lifecycle.c` (1/10/1000 cycles),
  `test_config_validation.c`, a central `aec_validate_config()` (rate
  whitelist, bounded fields) with saturating `ck_*` size arithmetic and
  misaligned-base rejection, and `docs_smoke.sh` (compiles the
  STATIC_MEMORY.md sample). `test_config_validation` 82 → 126 checks after the
  rate-relative HPF work.
- **Shared SIMD kernel layer** `audio_common/include/simd_kernels.h`
  (22 kernels, AArch64 NEON + always-compiled scalar twins, bitwise selftest
  incl. denormal/±0/inf and n=257 tails; `SIMD_KERNELS_FORCE_SCALAR` A/B
  knob). Bit-exactness rests on per-lane IEEE NEON ops + strict FMA discipline
  (explicit `fmaf` ↔ `vfmaq_f32`; plain mul+add never fused — consumer TUs
  must keep `-ffp-contract=off`); min/clip use compare+select (vminq/vmaxq
  diverge from the C ternary at ±0 ties).
- **`bin/bench_rtf`** (`make bench`; cross-compiles with
  `BACKEND=ne10 CC=<cross-gcc>`) and the clobber-permitted
  `fft_forward_scratch`/`fft_inverse_scratch` API, adopted at six dead-scratch
  rfft sites (input staging also skipped on NE10 there).
- **Build isolation (B01/B03)**: config-keyed final artifacts in
  `bin/<backend>-<cfg-sig>/` (full-coverage signature incl. AR/RANLIB and the
  resolved producer identity; `config.manifest` collision guard; `-MD -MP`
  header deps; `print-*` queries); `NO_STDIO=1` produces a stdio-free
  `libaec.a` (trace call sites compiled out, `aec_debug.o` excluded) and
  `audit-no-stdio` gates the delivered archive plus a stdio-free minimal-main
  executable — ne10 (board deliverable) is stdio-free end to end.
- **Toolchain discipline (P1-2)**: CC/CXX `-dumpmachine` coherence guard for
  `BACKEND=ne10` (a partial cross-toolchain override is now a hard error),
  with a `TOOLCHAIN_CHECK=0` opt-out that participates in the config
  signature.
- **publish v4**: `current` swapped via a rename(2)-atomic helper
  (`audio_common/tools/atomic_symlink_swap.c`, built with `HOSTCC` — BSD/macOS
  `mv` has no `-T` and follows symlink-to-dir destinations); the per-backend
  lock is now taken BEFORE the prerequisite build (concurrent same-config
  publishes can no longer race object/archive writes); `MANIFEST.txt` is
  deterministic (config + per-file SHA-256, byte-verified on republish —
  tamper detection); provenance (git commit/dirty/timestamp) moved to an
  append-only `ATTEST/` directory, one attestation per publish event;
  `DIST_ROOT=` redirects the release tree so isolation tests never touch real
  releases. Archive temp files are PID-suffixed (same-config concurrent build
  vs publish can't collide).
- **Publish provenance knobs**: `ALLOW_DIRTY_PUBLISH=1` (records
  `allow_dirty_publish=1` plus a `dirty_diff_sha256` — sha256 of `git diff
  --binary HEAD` — for whichever repo(s) are dirty); `ALLOW_UNTRACKED_PUBLISH=1`
  (records `untracked_tree_sha256` for this repo /
  `audio_common_untracked_tree_sha256` for the producer);
  `OBJ_ROOT`/`BIN_ROOT` placement knobs (not in `CFG_SIG`) so isolation tests
  can point the keyed obj/bin trees at a scratch directory and run
  tamper/injection scenarios against a real worktree build without ever
  touching the real `obj/`/`bin/` — default expansion (`obj`/`bin`) is
  byte-identical to the previous hardcoded paths, and `clean` now removes
  both; and `ATTEST_STAMP` (test-only) to override the UTC timestamp embedded
  in the attestation filename so isolation tests can force deterministic
  same-second collisions and prove the sequence-retry path.
- **Cross-repo provenance**: the attestation also records the audio_common
  *producer's* own commit/dirty state (`audio_common_git_commit` /
  `audio_common_git_dirty` / `audio_common_dirty_diff_sha256`) — this repo's
  release links against audio_common's archive, so "which source state
  produced this release" now spans both repos, not just this one. Attestation
  field order is otherwise unchanged and purely additive:
  `git_untracked`/`[untracked_tree_sha256]` follow
  `git_dirty`/`[dirty_diff_sha256]`;
  `audio_common_git_untracked`/`[audio_common_untracked_tree_sha256]` follow
  `audio_common_git_dirty`/`[audio_common_dirty_diff_sha256]`;
  `allow_untracked_publish` follows `allow_dirty_publish`.

### Changed

- **Detector wall-clock constants retimed to the live grid** (both ports).
  `detectors.py` / `detectors.c` carried ten constants frozen at this repo's
  legacy hop=160 @ 16 kHz (10.000 ms) grid, and neither file had a single
  retiming call site. None of the four shipping grids is 10 ms, so every one
  of them was mistimed everywhere: 8 kHz/128 and 16 kHz/256 (both 16.000 ms)
  ran authored durations **60 % long**, the 16 kHz/128 production default
  (8.000 ms) ran them **20 % short**, and 48 kHz/512 (10.667 ms) 6.7 % long.
  `ALPHA_CV`'s "TC ≈ 1 s" was in fact 1592 / 796 / 1592 / 1061 ms; it is now
  994.99 ms on all four. Retimed via the existing `aec3_scale.growth_rehop`
  (retention-convention EMAs) and `ms_to_hops` (hop-count gates); only the OLD
  term of each OLD/NEW pair is retimed, with NEW derived as `1 - OLD` so the
  pairs still sum to exactly 1 on every grid. Authoring grid confirmed
  numerically against the constants' own comments ("TC ≈ 1 s", "~4 s at
  100 fps", "TC~90ms"), not assumed.
  **Deliberately NOT retimed**: `CONV_FRAMES`, `CONV_ERLE_DB`, `DIV_ERLE_LIN`,
  `STATIONARY_CV2`, `ERL_CEILING_FLOOR`, `SAFETY_MARGIN`, and the
  `FilterPlateauDetector` ratio/threshold defaults. `CONV_FRAMES` is the
  interesting one: `update_convergence()` early-returns *without touching* its
  counter when far is inactive, warmup is unfinished, or near power is
  negligible, so non-qualifying hops are skipped rather than resetting it and
  its realised span is far-end-duty dependent. That makes it a consecutive-
  evidence count, not a duration — the same reading that spared
  `ne_recent_sustain=3` in the 2026-08-03 campaign. The rest are dimensionless
  level gates. `FilterPlateauDetector`'s two durations are retimed for
  consistency but the class is instantiated nowhere, so that retiming is
  unexercised; `CONSECUTIVE_REQUIRED` is left alone with its classification
  recorded as unresolved rather than guessed.

- **`SHADOW_FRAME_GATE`: C corrected 50 → 20, matching Python.** Commit
  `6cd995e` (2026-06-12) shortened the shadow-filter DT blind period
  500 ms → 200 ms in Python and never mirrored it to C, so the two ports
  disagreed for eight weeks. It survived because the one test that covers it
  was the orphaned `parity_detectors.c` above. The constant is now expressed as
  a duration (`DT_SHADOW_FRAME_GATE_MS = 200`) rather than a bare hop count, so
  the ports cannot drift apart per-grid either. **No audio effect in C**: the
  chain it feeds (`dt_from_shadow` → `AecResContext.shadow_dt`) has no reader
  anywhere outside the AEC's own equality test — verified by tree-wide grep.
  In Python the same signal does reach an audio path
  (`dt_from_shadow > 0.5` → `_ne_evidence` → `refined_usable`), so the C port
  is missing that consumer entirely. That gap is recorded, not fixed here.

- **16 kHz default signal grid** — `aec_config_defaults()` /
  `AecConfig.__post_init__` now default 16 kHz to the low-latency/low-compute
  256-point FFT / 128-sample hop grid (8 ms algorithmic delay) instead of the
  512/256 grid (16 ms) that M5 (multi-rate campaign, `d862a38`) had made the
  auto-default when it added 16 kHz/256 as a *selectable* option alongside the
  existing 512 grid. Only the auto-derived default changes; both 16 kHz grids
  remain fully supported and explicitly selectable.
  **Note on M5 itself**: `d862a38` (which introduced the 256/128 grid, the
  `aec3b_rate_cfg(sample_rate, fft_size)` dispatch table, and changed
  `aec_derive_dims()`'s hop derivation from a fixed `0.010 * sample_rate` to
  `fft_size / 2`) was not itself changelogged at the time it landed — its
  16 kHz *default* silently went from 10 ms to 16 ms hop as a side effect,
  with no entry here and no version bump. This entry covers both that
  already-shipped change and the default flip on top of it.
  **Verification — structural only, no perceptual/AECMOS bench this round**
  (explicit decision: the 800-case bench standard this file states above was
  not run for this entry; do not treat this as "regression-tested" for
  perceptual quality until that's done):
  - `test_rate_structural` extended with two new checks run at all 4 grids —
    impulse-through-**full AEC3 post-chain** (`enable_res=1`; the previous
    version of this test only ever ran the linear-only path) and
    `aec_init`/static-pool byte-equality vs `aec_create`/heap, also with the
    post-chain enabled. Nothing in the post-chain had been exercised
    end-to-end at 16 kHz/256 before this. 67/67 pass, including at 16000/256.
  - `test_static_aec` (default args, i.e. now exercising 256/128 at 16 kHz):
    static == dynamic, byte-equal, pool = 397,072 B.
  - `make selftest` (NEON vs scalar, `SK_HAVE_NEON=1`): all pass, unaffected
    (this change doesn't touch SIMD kernel code).
  **Measured static pool sizes** (BALANCED, all three presets identical —
  presets don't affect sizing): 8000/256 = 292,992 B; **16000/256 (new
  default) = 397,072 B**; 16000/512 (still selectable) = 543,040 B;
  48000/1024 = 1,233,680 B. (The 16000/512 figure supersedes
  `STATIC_MEMORY.md`'s previously-recorded 536,288 B, which predates this
  change and at least one other since; see that file's own note.)
- **One `FftHandle` per instance.** `aec_create`/`aec_init` now carve and
  construct exactly ONE `FftHandle` per instance (still exactly
  `fft_get_mem_size(fft)` bytes at the same `fft_size` every sub-module
  already used); the main filter and the shadow filter each borrow it instead
  of carving/owning a private handle, and `aec_post_fft` remains the single
  owner responsible for `fft_destroy`. Per-hop compute is unaffected (each
  sub-module still runs its own transforms against the shared handle, fully
  sequentially — shadow, then main, then the AEC3 post block, never
  interleaved, which this codebase's single-threaded synchronous call path
  already guaranteed). Safe because all three consumers run at the identical
  `fft_size` (derived once per instance in `aec_derive_dims`) and each
  `fft_forward`/`fft_inverse`/`fft_forward_scratch` call is a complete
  synchronous operation — no consumer retains a pointer into the handle's
  internal scratch across calls. Byte-equal verified: `aec_wav` renders across
  an 8-case stratified subset of the AEC Challenge corpus, KISS and NE10
  backends, before/after — MATCH on all 16. Full existing suite
  (counter-saturation, rate-structural, delay-reset, SIMD self-test,
  far-end-FFT-sharing regression test, context-only-processing regression
  test) re-run clean across BACKEND={kiss,ne10} x SIMD={0,1}.
- **Both pipelines rewired to the context-only seam**: mono
  `audio_pipeline.c`'s non-`aec_only` path and every lane of the 4ch
  `4aec_nr_res.c` wrapper now use
  `aec_process_context`/`aec_process_context_shared_far` instead of
  `aec_process`.
- **Static pool shrinks by exactly one hop-sized float buffer per instance**
  with the limiter gone (`aec_get_mem_size`, BALANCED, measured
  before/after): 16k/256 −528 B; 16k/512 −1,040 B; 48k/1024 −2,064 B.
- **Top-level (non-AEC3) hop-authored constants retimed to wall clock**
  (AecConfig, `aec.c`/`config.py`): `shadow_err_alpha` (0.80, main/shadow
  error-energy smoothing — EMA lag ~44.8 ms), `warmup_frames` (100 hops =
  1000 ms Kalman-Q warm-up window), `epc_hangover` (20 hops = 200 ms EPC-active
  hold), `ne_recent_hold` (150 hops = 1500 ms "near-end seen recently" hold),
  `filter_misadjustment_stable_frames` (30 hops = 300 ms required pre-fire
  stability window) and `filter_misadjustment_hangover_frames` (100 hops =
  1000 ms post-fire hold-off); also `EchoPathChangeDetector.EPV_FAST_TC` /
  `EPV_SLOW_TC` (0.98/0.999, `epc.py`/`epc_shadow.c` — fast/slow far-power
  EMAs feeding the EPV echo-path-change trigger; TC ≈ 495 ms / 9995 ms at the
  legacy grid). These were project-native constants, never sourced from AEC3
  at all, frozen at the legacy hop=160/sample_rate=16000 (10 ms) grid with
  zero rate conversion when the 16 kHz default flipped to 8 ms hop or at
  8/48 kHz — the same bug class as the AEC3-internal audit, just outside the
  AEC3-ported set that audit scoped to.
  **Left alone (genuine event counts, NOT durations — no rate conversion
  applies)**: `ne_recent_sustain` (3 — consecutive near-end-active hops
  required to ARM the `ne_recent_hold` timer above; the counted "event" is a
  per-hop occurrence, not an elapsed-time proxy, and it's a small fixed
  debounce, not a window). Each retimed-vs-left-alone call is justified by
  tracing every read site of the field, not just its name (`ne_recent_hold`
  and `ne_recent_sustain` sit right next to each other in config.py and read
  very similarly by name alone — only the usage sites disambiguate).
  Verified end-to-end: `make selftest` and `make test-counter-saturation`
  (12017/12017) unaffected; `parity_aec_e2e` PASS at all three rates
  (8/16/48 kHz, within existing tolerance — a genuine behavior change, not
  byte-equal: 16 kHz linear-path max diff 1.3e-5, post-chain max diff 1.9e-4,
  both well inside the 2.0e-2 gate); `aec_wav` runs end-to-end on a real
  800-case clip at both the default and `--fft-size 512`-overridden grid;
  Python smoke test (construct `AEC` + run 50 random hops) finite at all three
  grids.
- **AEC3 per-block/hop-count constants rescaled for the live `sample_rate`,
  Tier 1 — verified no-op at the current production grid (hop=160,
  sr=16000), only changes behavior at 8 kHz / 48 kHz**: ERL/ERLE hold +
  startup hops, `FilterAnalyzer`/`FilterDelay`/`FilteringQualityAnalyzer`
  convergence/consistency/adaptation thresholds, `InitialState` hops, PBFDKF
  leakage rates, `aec.c`'s poor-coarse/coarse-reset/leakage-div/stat-dt-hangover
  counters, `aec3_post.c`'s `y2_thr`, and reverb-decay wall-clock alignment in
  `residual_echo_estimator`. Each hand-verified to reproduce its old frozen
  constant exactly at hop=160/sr=16000 (e.g. `aec3_ms_to_hops(800.0, 160,
  16000) == 80 == ` the old `AEC_STATE_ACTIVE_RENDER_BLOCKS` literal). The
  conversion helpers are `aec3_ms_to_hops` / `aec3_blocks_to_hops` /
  `aec3_per_block_rate_to_per_hop` (`c_impl/src/aec3_scale.c` /
  `python/modules/aec3_scale.py`).
  **Explicitly held back, NOT included in this round**:
  `suppression_gain.py`'s `_LowNoiseRenderDetector` IIR (`0.9`/`0.1` →
  wall-clock-rescaled) and its C-side `suppression_gain.c` counterpart. This
  exact mechanism (`use_wallclock_low_noise_render_iir`-equivalent) was
  exposed as default-OFF research substrate in `[3.22.1]` and then removed
  entirely as dead code in a later cleanup (`"kept the literal 0.9 decay"`) —
  turning it back on in production a third time needs its own bench pass and
  sign-off, not a silent revival inside an unrelated rate-audit. Python's
  `use_wallclock_ema_alpha` kwarg is not passed from `orchestrator.py` (so
  `SuppressionGain`'s own `False` default applies); C's `suppression_gain.c`
  keeps the literal `0.9f`/`0.1f`. Tracked as a follow-up decision, not
  abandoned.
  Verified: `test_rate_structural` (67/67, all four grids),
  `test_counter_saturation` (12017/12017) with the held-back mechanism
  reverted to its pre-audit literal values. **Not yet run**: the 800-case
  AECMOS bench this repo's own rules require before Tier 2 items ship — treat
  this commit as a local, unpushed checkpoint, not a ship decision.
- **All production C converted to float32 end-to-end**, staged: (1) delay
  chain (`delay_aec3.c` biquads + scalar bookkeeping), (2) `aec.c`
  orchestrator scalars, (3) the post/state module chain (`aec3_post`,
  `aec_state`, `residual_echo_estimator`, `suppression_gain`,
  ERLE/reverb/stationarity), (4) mic-path HPF swapped to `audio_common`'s
  shared f32 platform HPF (DF2-transposed). `reverb_decay_estimator.c` is the
  sole remaining `double` file (dead code, no production caller). Also:
  de-stacked 13 large function-local `float[8192]`/`[4096]` arrays into the
  static pool, and a `/simplify` cleanup pass over the full diff.
- **Python bit-exact parity is retired repo-wide** — Python (fp64) is now the
  algorithm spec, C is the float32 implementation, Python↔C is
  tolerance-based (~−60 dB class), not 0/0. Gates: C-goldens
  (`test/parity_delay.c`/`gen_delay_c_golden.c`, `test/parity_aec_e2e.c`
  tolerance) plus staged checks vs the `fp64-baseline` tag — 60-case
  stratified AECMOS (worst Δ −0.021 echo, buckets ≤0.002), waveform drift
  median −95 dB, 1-hour soak stable. Static pool (BALANCED/16 kHz/52 ms) at
  that point: KISS 557,680 B (544.6 KB), NE10 519,232 B (507.1 KB), both
  static==dynamic byte-equal. Deployment model unchanged: host/reference =
  malloc + KISS (`make`, default); embedded = caller-pool + NE10
  (`make BACKEND=ne10`) — same main branch; NE10 vs KISS not bit-identical
  (pre-existing). `-ffp-contract=off`/`std=gnu99` retained.
- **Whole-repo per-bin/per-sample loop vectorization**, output
  **byte-identical throughout** (60-case render aggregate md5 unchanged on
  BOTH backends at every commit; static==dynamic byte-equal both backends;
  delay C-golden bit-exact; e2e tolerance unchanged). Converted: all
  complex-magnitude loops (~10.8k scaled-hypot calls/hop), echo_spec partition
  MAC + both filter W-updates, post-chain elementwise loops with fusion
  (coherence EMA+Γ² gate single pass; CNG N2 tracking; E2 select/gain apply),
  suppression-gain clips/min/final sqrt, numpy pairwise-sum trees (the numpy
  tree + BOTH tail-fold variants — probe-proven distinct at ±0 — now single
  shared implementations), `fft_power` (contraction made explicit as `fmaf`,
  objdump-verified identical codegen) and `fft_apply_gain` NEON in both FFT
  backends. Static pool **557,680 → 532,992 B KISS / 519,232 → 494,544 B
  NE10**; NE10 wrapper output staging memcpys removed. Perf: dev-box (arm64
  clang, auto-vectorizing) RTF flat-to-slightly-better — expected; the
  intrinsics make vectorization compiler-independent for the embedded gcc
  target. Measure on-target with `make bench`.
- **`main` now carries both memory models in one library** — `aec_create`
  (heap) and `aec_get_mem_size`/`aec_init` (single caller-owned pool, see
  `c_impl/STATIC_MEMORY.md`) — selected at runtime, mirroring the NR repo's
  single-branch model. Gates: consolidated `aec_wav` output byte-identical to
  the former branch build, and `test_static_aec` static == dynamic byte-equal.
- **Streaming FIFO rewritten to a provable SPSC ring (R02, replaces the first
  F09 fix)**: the previous atomic patch kept a shared `fifo_read` RMW cursor
  and was refuted by an interleaving proof (producer's drop-oldest advance
  could collide with the consumer's claimed slot at full; the consumer's
  underrun `memset` raced the producer's first write at empty). New protocol —
  Variant A′: `fifo_write`/`fifo_read` are monotonic unsigned sequences with
  exactly one writer each (producer / consumer), occupancy = `w − r`,
  `fifo_count` retired (kept at offset, always 0); overrun = **drop-new**
  (producer never writes when full, never touches the read cursor) +
  **consumer catch-up** (on observing a full ring the capture side skips to
  the freshest hop — staleness self-heals to ~1 hop after a burst, vs. the old
  drop-oldest's permanent ~320 ms lag); underrun uses a new immutable all-zero
  pool hop (`fifo_zero_ref`) — the ring is never written by the capture
  thread. Ownership proof in aec.h; every shared word has exactly one writer.
  `test_fifo_spsc.c` rewritten as a payload-integrity stress (per-hop sequence
  + full-hop bit-exact regeneration oracle via `far_hop`, producer/consumer
  jitter, bounded-staleness + gap accounting identities); `stream_sim.c` gains
  an exact-count overrun check and a catch-up check. Lockstep single-thread
  behaviour byte-identical (old vs new `--singlethread` dump `cmp`-equal;
  60-case render aggregates unchanged — offline `aec_process()` untouched).
  Pool +`ALIGN16(hop·4)` (16 k: KISS 538,320 B / NE10 534,192 B; per-rate
  table in STATIC_MEMORY.md). The earlier F09 shape (SPSC acquire/release
  atomics, layout unchanged, contract narrowed to one render + one capture
  thread, 100 k-hop concurrent stress + bookkeeping identity) is superseded by
  this rewrite.
- **FP policy (B04)**: `-ffp-contract=off` moved to last position with
  parse-time conflict rejection; AEC's own default codegen byte-identical —
  that campaign's one deliberate byte-change comes from audio_common+NR
  gaining the flag (they had none): new 60-case aggregates KISS `a2fc5d07…` /
  NE10 `44540201…` (supersede `652a2152…`/`09125432…`; median render delta
  −73 dB RMS, all parities within tolerance).
- **SIMD edge matrix (B05)**: n=0..17 sweeps, element-offset unaligned forms,
  canary guards, alias matrix, UBSan runner; finite corpus for the 12
  payload-unspecified kernels reconciled to classified comparison (UBSan run
  went fail → pass). 293,015 → 43,109,992 checks.
- **Build hygiene (R10/F12)**: object dirs are hash-keyed
  (`obj/<backend>-<cksum-sig>`) — no parse-time wipe, `make -n` is
  side-effect-free, different-flag builds coexist; cksum replaces shasum;
  parse-time config-stamp rebuild keying; `WERROR` knob.
- **Source-set identity + fresh archives (P1-4)**: the sorted source list is
  part of `CFG_SIG`; `libaec.a` is always built fresh (`$@.tmp` + `mv -f`).
- **publish v3 (P2)**: content-addressed immutable releases
  `dist/<backend>/<cfg_sig>-<content12>/`, idempotent republish, `LINK` in the
  signature, atomic `current` swap (GNU `mv -fT` + BSD fallback).
- **`ALLOW_DIRTY_PUBLISH` narrowed to tracked changes only**: the dirty check
  now uses `git status --porcelain -uno` (tracked-only) instead of full
  porcelain, since untracked content is now its own dimension — lumping the
  two together previously let two different untracked source states publish
  under the same `dirty_diff_sha256` (which only ever covered `git diff
  --binary HEAD`, i.e. tracked bytes). The FATAL wording changed accordingly
  (“uncommitted TRACKED changes”). Both checks — tracked dirty and untracked —
  are applied to BOTH this checkout and the audio_common producer checkout,
  and each FATAL names exactly which repo(s) and which dimension triggered the
  refusal.
- **Untracked-content provenance is fail-closed**: `untracked_tree_sha256` is
  a sha256 over sorted, COLLISION-FREE FIXED-FIELD records —
  `L <sha256(path)> <sha256(readlink output)>` for symlinks,
  `F <mode> <sha256(path)> <sha256(content)>` for regular files. Every
  variable-length value (the path itself, a symlink's target, a file's
  content) is itself hashed before being placed into the record, so records
  concatenate unambiguously — the original raw space-joined `L <path>
  <target>` encoding was collision-prone (`"a b"->"c"` and `"a"->"b c"`
  encoded identically). Any `stat`/`readlink`/`shasum` I/O failure downgrades
  that entry to an `X` record, which FATALs publish outright rather than ever
  recording an empty or partial hash — a hashing failure can never silently
  fall back to an incomplete-but-present `untracked_tree_sha256`. Two
  different untracked source states can never share a provenance record. An
  untracked path that is neither a regular file nor a symlink (an embedded git
  checkout, a fifo, …), or one that fails to hash for any reason, is always
  refused, naming the path and which repo it came from.
- **ATTEST one-event-one-file**: every publish event (including an idempotent
  republish) now writes exactly one NEW
  `attest-<utc>-<commit>[-dirty]-<seq>.txt`, installed via
  `atomic_symlink_swap.c`'s `--excl-install` mode (write-temp + `link(2)`, the
  atomic no-clobber equivalent of `O_CREAT|O_EXCL` — `link(2)` fails `EEXIST`
  if the name is already taken). A same-second name collision regenerates the
  event id with the next `<seq>` and retries. `git_commit` is now always the
  full 40-hex object id (round-5 used `git rev-parse --short`).
- **Documented 48 kHz matched-filter delay range corrected** from 203 ms to
  approximately 608 ms — the estimator first decimates 48 kHz to 16 kHz and
  rescales the result back to native samples; `test-delay-reset` now locks a
  300 ms acquisition case. (The multi-rate campaign's documented coverage was
  1216/608/~203 ms.)

### Removed

- **The peak-ratio output limiter** that ran as the last stage of
  `aec_process()`. It had no AEC3 counterpart — it was a local addition that
  applied a smoothed, one-hop-lagged time-domain gain
  (`final_output *= limiter_gain`, attack 0.3 / release 0.8) whenever the
  output frame's peak exceeded the microphone frame's peak. Applying broadband
  dynamics is not an echo canceller's job; clipping protection belongs to an
  AGC/DRC or output stage outside this library. Removed with it: Python
  `_limiter_gain` / `_limiter_near_lag`; C `limiter_gain` / `limiter_near_lag`
  / `has_limiter_lag` and their pool carve; the per-hop `near_peak`/`out_peak`
  scan and frame multiply; and the `limiter_gain` column from the
  `--debug-trace` CSV.
  **Explicitly NOT removed** (different mechanisms, all retained): the AEC3
  suppression-gain LF/HF limiters, input saturation detection, main/shadow
  filter divergence selection, and the final float→PCM hard-clip saturate in
  the WAV writer.
  **AI-AEC datasets and checkpoints are unaffected.**
  `materialize_linear_error()` reads `get_formed_output()`, which was always
  captured upstream of the limiter — verified byte-identical (same SHA-256 for
  both `linear_error` and `echo_estimate`) before and after this change, on a
  signal whose burst drives the output peak to 1.26. No regeneration needed.
- **Mono pipeline's dead "seam unavailable" fallback branch** — provably
  unreachable once `enable_res=0/return_res_context=1` is set unconditionally
  for every non-`aec_only` instance — removed rather than kept "just in case";
  and the 4ch wrapper's now-unused `lane_out` staging buffer.
- **Per-hop `W_all`/`X_buf_all` snapshot copies** (2×~12 KB/hop) — post reads
  filter state via const pointers instead.
- **`parity_hpf.c` and `gen_hpf_golden.py`**, with the mic-path HPF move to
  audio_common's shared f32 platform HPF.
- **The `feature/static-memory` branch**, and the local
  `c_impl/{src,include}/hpf.{c,h}` (moved to `audio_common` as `hpf_f64`; pure
  rename, output byte-identical, `parity_hpf` golden still bit-exact at 0
  error). This removes the last local copy of an `audio_common` component.
- **Hardcoded ne10 `-lc++`** (the C++ runtime comes from the `LINK=$(CXX)`
  driver; GNU/Linux uses libstdc++ automatically).

### Fixed

- **`pbfdaf_free`'s static-path double-free**: it no longer calls
  `fft_destroy` (it never owns the handle it borrowed); calling it there as
  well would double-free the NE10 backend's twiddle-config allocation every
  borrower now shares. `pbfdaf_reset` is unaffected (it never touched
  `p->fft`).
- **`pbfdaf_reset` now also clears `precomputed_far_spec`** — latent but never
  triggered before far-FFT sharing existed (the field was always NULL by the
  time any reset could observe it, since it was previously set-and-consumed
  purely within a single internal call); with cross-instance sharing, a caller
  resetting an instance between setting the pointer and that instance's own
  process call could otherwise leave a stale/potentially-freed pointer on the
  struct.
- **AI-AEC `dataset_gen` ch5 read the limiter-processed output.**
  `Audio_ALG/AIAEC/dataset_gen/linear_aec.py`'s `ch5` channel (the dataset's
  linear-AEC-error reference signal) read `process()`'s output instead of the
  pre-limiter formed hop — a discontinuity whenever the limiter's gain
  excursion crossed 1.0 mid-utterance, visible as a vertical line in ch5's
  spectrogram. Fixed by switching `LinearAecProcessor` to the
  `return_formed_output` seam.
- **Retention-vs-new-sample EMA convention bug, caught mid-implementation**:
  `shadow_err_alpha` and `EPV_FAST_TC`/`EPV_SLOW_TC` all use the update
  `x <- c*x_old + (1-c)*new` — the constant `c` multiplies the OLD state
  directly (a RETENTION convention). This is the OPPOSITE of AEC3's own
  convention that `aec3_per_block_ema_alpha_to_per_hop` assumes
  (`x <- (1-a)*x + a*new`, where `a` multiplies the NEW sample). Applying the
  existing AEC3-style helper to a retention-convention constant silently
  retimes it **backwards** (verified: it produced smaller per-hop retention at
  a *shorter* hop, when a shorter hop needs *larger* per-hop retention to
  match the same wall-clock decay). A new, correctly-generalized helper was
  added instead of reusing the mismatched one.
- **Retiming point moved from `aec_config_defaults()` to `aec_carve()`.**
  Baking the AecConfig-level retiming into `aec_config_defaults()` (the
  natural-looking spot, alongside `filter_length`'s existing sr-based
  derivation) is wrong — `aec_wav.c`'s `--fft-size` flag (and
  `test_rate_structural.c`'s own alternate-grid selection) overrides
  `cfg.fft_size` *after* `aec_config_from_preset()` returns, so retiming
  against the default-grid hop at that point freezes the wrong grid the moment
  a caller picks the non-default 16 kHz/512 grid. Retiming now happens once in
  `aec_carve()` (construction time, when `aec_derive_dims()` has already
  re-resolved `hop` from the FINAL `cfg->fft_size`), writing into `a->cfg`
  (the carved instance's own copy) — the caller's original `AecConfig` is left
  untouched. Python has no equivalent post-construction grid-override call
  site (`__post_init__` runs once, immediately, and no code in this repo
  mutates `frame_size`/`hop_size` afterward), so `AecConfig.__post_init__`
  remains the correct, single retiming point on that side. `epc_init()`'s two
  in-repo call sites are `aec.c`'s `aec_carve()` (passes the real carve-time
  grid) and `test_counter_saturation.c`'s `section_epc_shadow` (passes the
  legacy 160/16000 — that test only exercises hangover countdown, not EPV);
  the retimed EPV values are computed once at `epc_init()` time,
  config-slice semantics, not touched by `epc_reset()`. Python's sole call
  site (`orchestrator.py`) now passes them explicitly.
- **Native 8 kHz delay-estimator feed rate.** `delay_aec3_init()`
  (`c_impl/src/delay_aec3.c`) was hardcoding the internal delay estimator's
  feed rate to 16000 regardless of the AEC's actual configured `sample_rate`.
  `da_estimator_init()`/`da_clockdrift_init()` derive real-world timing
  constants from that rate (the ~30 s clockdrift `stability_reset_hops` and
  the ~500 ms `consistent_estimate_threshold`, both
  `seconds * sample_rate / DA_AEC3_BLOCK_SIZE`) — passing the wrong rate
  distorts their real elapsed time, not just a label. 16 kHz was
  coincidentally correct; 48 kHz was also correct, but only because the
  render/capture pair is resampled to a 16 kHz-equivalent stream before
  reaching the estimator. Now passes the true feed rate (native `sample_rate`
  at 8/16 kHz, 16000 only when the 48 kHz sidechain is active). Verified:
  `test_rate_structural` (67/67, all four grids), `test_counter_saturation`
  (12017/12017, incl. `section_delay_aec3`), an isolated old-vs-new
  before/after comparison at native 8 kHz, and `test_static_aec`
  static==dynamic at all three rates (8000: 193,280 B pool; 48000:
  byte-equal).
- **Python 48 kHz anti-alias sidechain was missing**, making Python and C
  structurally different at 48 kHz. `EchoPathDelayEstimator`
  (`python/modules/delay/echo_path_delay_estimator.py`) now ports the C-side
  `DaResample48` 48 kHz→16 kHz sidechain (added 2026-08-02, previously flagged
  C-only). A `parity_aec_e2e` investigation found the asymmetry caused Python
  and C to lock onto *different delays at different hops* at 48 kHz (Python
  hop 68/delay 1024 vs C hop 100/delay 960) — not a numerical-precision drift
  but a structural mismatch, since Python fed the matched filter raw, aliased
  48 kHz samples in 1.33 ms inner blocks instead of the intended 4 ms. Added
  `_Resample48` (same order-7 elliptic 4-SOS-section coefficients as
  `DA_RESAMPLE48_B`/`A` in C; `scipy.signal.sosfilt`'s default recurrence is
  the identical Direct-Form-II-Transposed form `da_biquad_process1` implements
  — verified sample-for-sample against the C filter to float32 precision on a
  test signal). `ClockdriftDetector`/`MatchedFilterLagAggregator`'s
  `sample_rate` parameter is now the *effective* inner-block rate (16000 when
  the sidechain is active) rather than the raw native rate, mirroring the C
  fix. `parity_aec_e2e` 48 kHz max diff: 4.596e-01 (FAIL, tol 1.0e-01) →
  3.873e-02 (PASS) — back in the same class as the originally-measured 6.4e-2
  baseline that calibrated the 48 kHz tolerance. 16 kHz/8 kHz outputs
  unchanged (rate_factor=1 path is a no-op).
- **`python/diag/gen_aec_e2e_golden.py` hardcoded `hop = sr*10//1000`**
  (10 ms), which diverged from the constructed `AecConfig`'s real hop once
  16 kHz's default grid moved to 256/128 — regenerating a golden at the new
  default crashed (`ValueError: could not broadcast ... (160,) into (128,)`).
  Fixed to read `hop_size` back from the constructed config. This is why the
  48 kHz asymmetry above went undetected: the E2E gate had been silently
  unable to run at any rate since the grid default changed.
- **`delay_aec3_reset()` / `EchoPathDelayEstimator.reset()` now clear the full
  signal chain** — previously left the inner decimators/ring/pending-sample
  state stale after the 48 kHz sidechain's own reset (a mixed fresh/stale
  reset). `c_impl/test/test_delay_reset.c` + `python/tests/test_delay_reset.py`.
- **`erle_startup_hops`/`erl_startup_hops`'s `200`-as-sentinel collision**:
  `None`/`-1` now means "auto"; an explicit `200` request could previously
  never be expressed, since it was indistinguishable from the auto default.
  `python/tests/test_aec_state_startup_hops.py`.
- **C `RenderActivity` now matches Python on activation**: the first active
  render hop initializes the envelope but is not classified stationary until a
  second observation exists. A three-state regression covers initial
  activation, the second hop, and activation after silence.
- **8 kHz taps OOB (F01)**: 880-alloc/960-read — the external review's
  heap≠pool SHA divergence — fixed; delay-estimator geometry now mirrors the
  Python spec (fixed /4).
- **Zero-heap (F02/F08)**: NE10 twiddle configs are carved from the caller
  pool (audio_common vendored patch P0001); `aec_init→destroy` makes zero
  allocator calls on both backends (`test_zero_heap_aec.c`, hook-verified);
  destroy is a true idempotent no-op for pool instances.
- **Lifecycle leaks (F03/F04)**: `aec_create` arena-fied onto the shared
  `aec_carve()` — 87 leaks / 184 KB per lifecycle → 0, and −316 lines of
  hand-mirrored duplication; pbfd* inits return status; single-path rollback.
- **WAV ingress (F06)**: hardened single reader in audio_common (shim here);
  fmt/format/bounds/odd-chunk/EOF checks, non-finite float ingress sanitized +
  counted.
- **SIMD NaN divergence (F10/R07)**: the cabs helper is NaN-exact vs scalar
  (`vmaxq`/`vminq` AND `vabsq` were all NaN-divergent); NaN corpora added to
  the selftests; fast_math domain guards (fast_exp NaN OOB-LUT fixed). The
  soft-warn path in `simd_selftest_aec.c` is a classified assertion now —
  every element must be bit-equal or both-NaN (payload unspecified: multi-NaN
  reduction tie-breaks are out of contract); anything else exits non-zero. All
  60 historical soft mismatches (265 elements) classify as both-NaN; no kernel
  needed fixing. Special-value contract table + pinned selftest values live in
  audio_common (`fast_math.h`).
- **Docs (F19)**: real 3-param `aec_init` everywhere; ownership truth per
  backend.
- **Float-WAV outer RIFF size** (audio_common, R01) — that campaign's only
  deliberate byte-change: float-WAV headers gain the correct outer RIFF size.
  New 60-case aggregates: KISS `652a2152…` / NE10 `09125432…` (payload
  verified byte-identical per file; PCM16/pipeline/NR outputs unchanged).
- **Stale `libaec.a` archive members** — a removed source can no longer leave
  one behind.
- **publish v3's `current` swap followed the existing `current` symlink into
  the release dir** (a round-3-latent republish bug); fixed by the atomic swap.
- **Missing-`current` window and concurrent same-config publish race** —
  closed by publish v4's rename(2)-atomic helper and by taking the per-backend
  lock before the prerequisite build.
- **`make -n/-q/-t publish` was not side-effect-free.** Round-5's driver
  mentioned `$(MAKE)`, so GNU make ran it for real even under `-n`,
  transiently mkdir'ing `dist/` and taking/releasing the real lock on every
  dry run; a MAKEFLAGS word-scan now branches BEFORE `$(DIST_ROOT)` is created
  or the lock is taken (`-n`/`-t` recurse into a print-only path, `-q` exits 1
  per question-mode's "needs updating" semantics). Round-6 then documented all
  three as side-effect-free, but that was still not true for `-t` — the
  round-6 driver recursed into `_publish_impl` under `-t` exactly as under
  `-n`, and GNU make's standard touch semantics on that recursive recipe chain
  bumped the mtimes of the real `libaec.a`/`aec_wav`/`bench_rtf` (and, via the
  `AC_LIB` dispatch, audio_common's own archive) — a genuine write for a
  target that is supposed to be a phony action, not a build product. `-t` is
  now an explicit no-op: it prints a one-line note and exits 0 without
  recursing. All three flags are now verifiably zero-write.
- **Combined dry-run flags** (e.g. `make -nt publish`) are now handled
  `-t`-FIRST: the driver's `MAKEFLAGS` word-scan checks `has_t` before
  `has_q`/`has_n`, because in a combined invocation the old `-n`-first
  ordering handed the recursive child `make` both flags at once, and GNU make
  really applied touch semantics to that recursive chain regardless
  (reproduced) — the same real-write bug the `-t`-alone fix addressed, just
  reachable through a different flag combination. Any `t` now wins
  unconditionally over `n`/`q`.
- **Same-second republish silently clobbered the prior attestation** (round-5
  used `mv -f`); an existing attestation is never overwritten now.
- **Identity-less checkout could previously publish**: the prior shape folded
  an `unknown` identity into the same `dirty`/`ac_dirty` violation flag that
  `ALLOW_DIRTY_PUBLISH=1` short-circuits, so passing that knob alone could let
  a non-git (or an unresolved-`AC_DIR`) tree publish despite having no real
  provenance to record. A checkout — this repo OR the resolved audio_common
  producer — for which `git rev-parse HEAD` fails is now FATAL immediately,
  checked before `dirty`/`untracked` are even computed for that repo, and
  neither knob admits it.
- **Latent `hpf_init`/`hpf_process`/`hpf_reset` symbol collision** with
  audio_common's f32 platform HPF (different signatures, ABI-incompatible),
  killed by the `hpf_f64` move.

### Security

- The rewritten streaming FIFO is **TSan-clean by construction, with zero
  suppressions** — every shared word has exactly one writer, and the ownership
  proof lives in `aec.h`.

### Known limitations

These are stated so a reader does not mistake "all tests green" for "validated
at every grid". They are the reason this release carries an `rc` marker.

- **48 kHz has structural evidence only.** Both retiming batches (16), (17)
  were A/B-validated on the two 16 kHz grids and verified numerically stable at
  48 kHz — 1875 hops of far-only / double-talk / near-only / echo-path change /
  mid-stream `aec_reset()`, zero non-finite samples, finite `erl_estimate` and
  `saturation_level` throughout. That proves the 48 kHz path does not break; it
  does **not** validate 48 kHz tuning. No native 48 kHz far-end / double-talk /
  near-end material exists in this repo, and upsampled 16 kHz material cannot
  substitute — it carries no energy above 8 kHz, which is precisely the band a
  48 kHz grid adds. A formal 48 kHz release needs native recordings.

- **The 800-case bench has not been re-run for this release.** The retiming
  batches used a 90-case blind subset. Breaking items (8) and (9), plus the
  16 kHz default-grid change under "Changed", alter production output and
  still require the full release benchmark.

- **Wall-clock retiming is validated as timing, not as tuning.** Every constant
  now covers the same wall-clock span at every grid, which is the property the
  authoring comments always claimed. Whether that span is the *best* value at a
  non-16 kHz grid is a separate question that no test here answers.

---

## [3.24.1] — 2026-06-25 — Warm tap-transfer on delay acquisition (cold-start "vertical line" fix)

A no-PA-path BALANCED change. At no-pre-align cold start the far-end is silent
until ~1 s, so the online matched-filter delay estimator cannot lock until then;
when it does (e.g. 2uGeP `..._farend_singletalk`, frame 114 / t=1.14 s, delay 0→448)
the ring buffer realigns and the PBFDKF filter was **fully reset** — discarding
the ~12 dB cancellation it had already built at the cold alignment. The exposed
echo is a **bright broadband vertical line** at the lock instant (clearest at the
linear stage; the RES/NR post-filter suppresses ~25 dB of it downstream, which is
why it reads as a faint haze in the final output). The reset is the filter's
destructive reaction to a **correct** realign — the delay estimate itself is right
(true delay ≈503, estimated 448 = within half a block; 1.14 s lock is the
conservative `is_solid + ≥3 updates` gate, not an error).

**Fix — `delay_acquire_warm_transfer` (default ON).** Instead of zeroing the
filter, shift the learned impulse response LEFT by the acquired delay (time-domain
`irfft → roll → rfft` per partition, exact across partition boundaries) so the
cancellation survives the realign. Gated to the line condition: the filter was
genuinely cancelling (recent inst-ERLE peak > `delay_acquire_inst_erle_db` = 4 dB
— `erle_windowed` lags ~0 here so it cannot be used) AND the delay fits the tap
reach (`n_partitions · hop`). Outside the gate it falls through to the existing
reset path unchanged. Unlike AEC3 — which dodges the line via
`SetInitialState(true)` + routing raw mic during re-convergence (wrong for an
NR/NN back-end that cannot cancel echo from raw mic) — this **keeps our strong
linear filter**, which is the asset for a linear→NN/NR pipeline.

**Validation (no-PA / `NO_PREALIGN=1`, the production-realistic path).** Fires on
**117 / 600** FS+DT cases (delay < tap reach AND filter cancelling); non-firing
cases are byte-equal to the zero-reset path. Per-case AECMOS A/B on the firing
population (non-firing Δ ≡ 0, so this is the exact full-corpus delta):

| Bucket | AEC stage (enable_res) ΔFULL | NR pipeline (Audio_ALG) ΔFULL |
|---|---|---|
| FS_static | echo +0.003 / deg −0.000 | echo +0.0015 / deg −0.000 |
| FS_movement | echo +0.001 / deg −0.000 | echo +0.0022 / deg −0.000 |
| DT_static | echo −0.001 / **deg +0.029** | echo +0.0028 / **deg +0.019** |
| DT_movement | echo +0.007 / deg +0.004 | echo +0.0073 / deg +0.009 |
| NE | byte-equal (warm never fires) | byte-equal |

Every bucket echo+deg neutral-to-up on both the AEC's own RES path **and** the
real `AEC linear → NR → RES(min, ne_floor=0.4)` pipeline; no ship bar threatened;
NE untouched. Mechanism: removing the residual-echo line at the linear stage means
RES/NR need not suppress as hard → less near-end damage → DT deg up. 2uGeP line
level −12.3 → −23.9 dB with steady-state cancellation preserved (12.2 vs 11.9).

**Byte-equal under the legacy pre-align bench** (delay-est OFF for non-movement →
no acquisition realign → warm never fires), so the default-config 800-case AECMOS
numbers for `[3.24.0]` are unchanged. `delay_acquire_protect_inst_erle` (option A,
block the realign entirely) was A/B-rejected — echo cost not cleanly gateable —
and is kept default-OFF for reference.

---

## [3.24.0] — 2026-06-22 — AEC3 round-robin TD constraint (FFT cut + echo gain) + DT-floor re-tune

A BALANCED algorithm change from an FFT/IFFT audit of the AEC→NR→RES path. The
time-domain gradient constraint (the per-partition `irfft→window→rfft` that zeros
each filter partition's non-causal tail) was applied to **all** partitions
**every hop** on both the main (PBFDKF) and shadow (PBFDAF) filters — whereas
AEC3 constrains **one partition per hop, round-robin**
(`adaptive_fir_filter.cc:686-689`). That constraint was ~70% of the integrated
per-hop FFT cost (24 of ~34 ops/hop).

**(1) Round-robin constraint (`constraint_round_robin`, default ON).** Constrain
one partition per hop, cycling (`partition_to_constrain`), on each filter →
constraint FFTs 24→4 ops/hop (**~58% of the integrated per-hop FFTs removed**).
The old all-partitions-every-hop form was *over-constraining* (the float32
irfft/rfft round-trip + boundary window repeatedly nibbled freshly-adapted taps),
so the lazier round-robin also **deepens linear convergence → more echo
cancelled**. Verified genuine (not a near-end trade): FS buckets (no near present)
gain echo with deg pinned at the 4.999 ceiling, and matched-echo shows the FS gain
is 3–9× the far-active-floor knob's at equal deg cost — round-robin is *not*
dominated by the floor knob.

**(2) DT-floor re-tune (`min_gain_floor_dt_db` −20 → −16).** Round-robin's deeper
convergence frees FS-echo headroom; spending it on the DT-only min-gain floor
neutralises round-robin's standalone −0.031 DT-deg cost.

800-case no-PA (balanced / fl=832 / cng), vs the full-constraint 3.23.0 baseline:
echo up in every far-active bucket (FS_static +0.029, FS_movement +0.022,
DT_static +0.014, DT_movement +0.027), **DT deg held** (DT_static 2.074→2.076,
DT_movement 2.140→2.136), NE flat (deg −0.009); all four ship bars pass. C port
mirrors both filters (`pbfdkf.c` constraint sites gate on `partition_to_constrain`;
`aec.c` enables on main + shadow); `parity_aec_e2e` PASSes all 3 presets with
*tighter* tolerance than before (linear max 9.8e-7 / out 2.0e-5 — round-robin does
6× fewer FFT round-trips, so less float32 drift accumulates).

---

## [3.24.0-dev] — 2026-06-20 — C FFT backend: pocketfft → KISS (NR-shared, float32)

Landed after the 3.23.0 tag and shipped inside 3.24.0; it carried no version
bump of its own because `__version__` tracks the Python reference, which this
did not touch. Retitled from `[Unreleased]` 2026-08-06 — it has been released
for over a month, and a stale `[Unreleased]` heading in the middle of a
released history is unreadable.

**C-only change; Python reference + `__version__` (3.23.0) unchanged.** The C FFT
backend is swapped from the vendored numpy pocketfft (fp64, numpy-bit-exact) to
**KISS FFT (float32)** — vendored `c_impl/lib/kiss_fft/`, shared with the NR repo
— with **NE10 (ARM NEON) opt-in** via `make NE10_DIR=...` (mirrors NR's
Makefile). The KISS static-memory path is fully heap-free (cfg placed in the
caller pool via `kiss_fft_alloc`).

Consequence: C↔Python parity drops from strict 0/0 to **float32 precision** (KISS
float32 ≠ numpy fp64 `np.fft`). Measured end-to-end (kiss-C vs Python, balanced
doubletalk): **correlation 0.99999958, RMS error ≈ −60 dB below signal,
per-sample max ~6e-3** over 4186 recursive hops — inaudible, NR's shipped
alignment standard. The **non-FFT C logic remains bit-exact** (the v3.23.0
verification stands); only the FFT layer carries float32 tolerance.
`parity_aec_e2e.c` now asserts a 2e-2 float32 tolerance instead of 0/0;
`fft_pocketfft.c` + `lib/pocketfft/` removed.

---

## [3.23.0] — 2026-06-20 — no-PA delay fix + DT-deg recovery + Python↔C bit-exact completion

The first BALANCED algorithm change since 3.22.x. Two production changes plus a
Python↔C parity completion. Bench is now reported **without pre-alignment**
(online matched-filter self-align, production-faithful).

**(1) Matched-filter pre-echo fix (no-PA delay).** The matched-filter
`accumulated_error` was binned by capture-sample index (`i // 4`), so only bins
0–3 ever filled and pre-echo always collapsed to ~0. With `detect_pre_echo`
default-True, that zero pre-echo *overrode* the correct highest-peak delay →
the no-PA online delay was systematically pulled back toward 0 (pre-align was
immune, masking the bug). Fixed to the AEC3 cumsum prefix-error form
(`matched_filter.cc:516-524`). Restores online delay acquisition; FS/DT echo up
across every bucket. C port `delay_aec3.c` in sync.

**(2) DT-deg recovery stack (default-ON).** Online delay acquisition's
filter-reset cost lands on double-talk. Two energy-gated levers (`_ne_recent_frames`,
not coherence): `dt_aware_recovery_soft` (soft acquire-reset — keep the converged
filter) + `dt_aware_res_floor_enabled` with `min_gain_floor_dt_db=-20` (DT-gated
RES min-gain floor lift toward near-end protection). DT deg recovered above the
pre-align reference while all ship bars hold.

No-PA 800-case BALANCED (online self-align): FS_static 3.544 / FS_movement 3.519
/ DT_static 4.218·2.074 / DT_movement 4.114·2.140 / NE 4.021 — four ship bars met
(FS echo>3.5, DT echo>4, DT deg>2.0, NE deg>=4).

**(3) Python↔C bit-exact completion.** Every module parity test + the end-to-end
`parity_aec_e2e` now pass with **0 mismatches under `-DUSE_STANDARD_MATH`** across
all three presets (4186 hops). Production `fast_math.h` (approximate
`fast_exp`/`fast_sqrt`) is the only residual (~1e-5..1e-4 in exp/sqrt stages; the
linear/PBFDKF path stays bit-exact under both backends). The parity tests are
logic checks — build with `-DUSE_STANDARD_MATH`; that flag yields true bit-exact.

Four real production-C correctness bugs were found and fixed so the C matches the
validated Python source-of-truth (not test-only): `filter_analyzer` read the gain
peak from the raw filter taps instead of the high-pass-filtered taps;
`residual_echo_estimator` gated its nonlinear noise gate on `transparent_mode`
instead of `use_stationarity_properties` (collapsing nonlinear R² to 0);
`aec.c` was missing the orchestrator's Track-F sustained-leakage gate in the
PBFDKF `H_error` refresh (linear output diverged from hop 79); `suppression_gain_init`
did not zero `dt_protect_active`. Several parity tests/golden-gens also had infra
gaps (unset `lf_clamp_bin`, missing DT-floor replay, a stale gen kwarg, a missing
`hpf` golden-gen) — those are test-only and do not affect production.

## [3.22.5] — 2026-06-07 — streaming C API + release cleanup (BALANCED byte-equal)

Streaming C API + dead-code removal + research-arc closures. **BALANCED algorithm
unchanged** — 800-case AECMOS scores are identical to 3.22.4 (FS_static 3.576 /
DT_static 4.201·2.156 / NE 4.047); the bump is for the new public API + hygiene.
The retired full release report and 3-way AEC3/Speex comparison remain
recoverable from Git history.

**Release cleanup (both impls, bit-exact)**
- Removed 10 default-OFF research flags (none enabled by any preset) + the opt-in
  DTD detector subsystem (`dtd.py`/`dtd.c`) + `NearendSpp`: **−1312 lines Python,
  −423 lines C**. Python byte-equal (27/27 wavs ×3 presets, 25/25 tests); Python↔C
  **bit-exact** (peak|Δ|=0, ±CNG, all 3 presets); per-module + e2e golden harnesses pass.
- Python CLI now exposes all three presets (`--preset gentle|balanced|aggressive`).
- ~25 GB of /tmp build scratch (webrtc checkout, aec3 dumps) cleared; docs curated.

**Streaming render/capture API** ([c_impl/include/aec.h](c_impl/include/aec.h),

**Streaming render/capture API** ([c_impl/include/aec.h](c_impl/include/aec.h),
[c_impl/src/aec.c](c_impl/src/aec.c); commit `632de7a`)
- `aec_analyze_render()` / `aec_process_capture()` decouple far-end (render) and mic
  (capture) for real-time pipelines where they arrive on separate calls/threads, not
  necessarily 1:1. A 320 ms render-hop FIFO absorbs call-scheduling jitter (**NOT**
  echo delay — that stays in `ref_ring`):
  - underrun (capture w/ empty FIFO) → process with silent render + `AEC_BUF_RENDER_UNDERRUN`
  - overrun (render past capacity) → drop oldest + `AEC_BUF_RENDER_OVERRUN`
- Lockstep (one `analyze_render` then one `process_capture`) is FIFO pass-through →
  **byte-identical to `aec_process()`**. Verified by [c_impl/test/stream_sim.c](c_impl/test/stream_sim.c):
  lockstep 0/400 hops differ; overrun + underrun detection/recovery PASS.

**Research-arc closure (no production change)**
- `aec_record` 6–7s breath/fricative closed as **two interleaved Pareto trades, not a
  single HF bug**: the linear stage already preserves the breath better than Speex
  (NORES −0.20 vs Speex −1.03; the −4.53 dB cut is 100 % the RES); the audible 4–6k
  loss is echo-dominated (`near_fr` 0.13, reference-blind); a per-frame near-priority
  mask separates 1–3k window-averaged but leaks ~12 % of FS far-active frames. BALANCED
  stays byte-equal; `gentle` is the opt-in speech-preserving operating point, not a fix.
  See docs/breath_aec_record_6_7s_closeout_2026_06_07.md.
- Removed stale top-level diag scripts (`diag_enr_trace.py`, `spp_step0_diag.py`).

## [3.22.4] — 2026-06-03 — three Pareto presets + finalization hygiene

Finalization of the v3.22 DSP arc. The DT-deg gap vs AEC2 is now a **proven
single-channel DSP Pareto wall** — a `coh_gain_floor` sweep showed even the
weakest strength (0.25) trades +1.22 DT-deg for breaking all four echo bars
(FS_static −0.39 / FS_movement −0.30 / DT −0.40…0.50). There is no free DT-deg
DSP win; the honest response is to **expose the trade as a strength axis** rather
than hide it. BALANCED is **byte-equal** to 3.22.3 (`__version__` bump is for the
new presets + hygiene, not a BALANCED behaviour change).

**Three Pareto presets on a single residual-echo strength knob**
(`min_gain_floor_far_active_db`, the far-active min-gain floor):

| preset | floor | NE deg | DT_s e/d | DT_m e/d | FS_s echo | FS_m echo |
|---|---|---|---|---|---|---|
| **gentle** | −20 dB | 4.052 | 4.004/2.305 | 3.893/2.387 | 3.433 | 3.388 |
| **balanced** | −28 dB | 4.047 | 4.201/2.156 | 4.082/2.228 | 3.576 | 3.512 |
| **aggressive** | −38 dB | 4.046 | 4.370/2.049 | 4.249/2.117 | 3.664 | 3.590 |

- `gentle` (near-priority): DT_static deg **2.305 == AEC2 (2.304)**; NE held ≥4;
  FS echo deliberately below balanced's 3.5 bar (above the ≥3.2 sanity floor) — the
  accepted near-preservation Pareto trade.
- `aggressive` (echo-priority): **beats AEC2 on DT echo (4.370>4.331) and FS echo
  (3.664>3.457)**, deg still >2.0 and well above AEC3; NE held.
- Monotonic ±10 dB spread; gentle/aggressive differ from balanced **only** in the
  floor (verified). `--preset gentle` byte-identical to `AEC_FAR_ACTIVE_FLOOR_DB=-20`.
  Audio: gentle retains more residual than aggressive in 6/6 DT cases.
- Bench: full-800 `preset / fl=832 / cng`. Pareto-matched per [[feedback_aecmos_pareto_comparison]].

**Hygiene (byte-equal, 48/48 `_ours` + `_ours_nores`):**
- Retired the `enable_highpass_ref` dead flag + its 3 orchestrator branches
  (ref-path HPF was retired in v3.19; `_hp_ref` always None).
- Deduplicated a redundant `_inst_erle_smooth` `__init__` assignment.
- Removed scratch (`cmp_cohf_sweep.py` + 2.4 GB of stale render dirs).
- **Deferred:** DTD subsystem removal (dead-but-gated; safe excision entangles
  `_compute_mu_scale` + the post-NlmsFilter LMS branch — reserved for a dedicated
  isolated change, not a cleanup pass).

**Diagnostics:** confirmed the per-frame trace surface
`run_one_case.py --diag-csv [--trace-aec-state]` (45-col CSV: ERLE/gain/R²/ENR/
dt-conf/delay/convergence/usable_linear) — the schema the C port mirrors.

**C port — end-to-end BIT-EXACT to `python/aec.py`.** The `c_impl/` production
port now matches the Python reference byte-for-byte: per-hop golden 0 mismatches
over the full doubletalk case (linear residual + final output), all three
presets; full CLI `wav→wav` 0/669920 fp32 sample mismatches. The v3.10 pipeline
(legacy 9-stage `ResFilter` + GCC-PHAT/PAR delay + P-denominator Kalman) was
rewritten to the v3.22 AEC3 chain: vendored numpy pocketfft FFT, PBFDKF per-bin
H_error filter, AEC3 matched-filter delay (`delay_aec3`), FormLinearFilterOutput,
the full `_aec3_post` orchestration (`aec3_post_run`: AecState + ResidualEcho­
Estimator + SuppressionGain + CNG/OLA), and the 21-step `aec_process`. Bit-exactness
required matching three numpy-on-arm64 idioms (`np.abs(c64)**2` = SIMD
scaled-hypot-FMA squared, complex64×complex64 multiply uses FMA, EMA `(1-α)` is a
double subtraction cast to f32) — `-ffp-contract=off` is mandatory. Per-module
golden tests (`test/parity_*.c`); end-to-end gate `test/parity_aec_e2e.c`. A
heap-free **static-memory** pool variant (`aec_init`/`aec_get_mem_size`,
byte-equal) ships in-tree — see `c_impl/STATIC_MEMORY.md`. Opt-in
per-frame CSV trace via `aec_wav --debug-trace <path>` (audio-passive).

## [3.22.3] — 2026-06-03 — isolated parity/correctness candidates (P0 audit; AECMOS-neutral)

Adjudicated the independent source-audit findings as **isolated parity/correctness
candidates** — each its own 800-case A/B + per-case **energy audio-proof**, gated
if it regressed or only reshaped without benefit. Output changes vs 3.22.2
(NOT byte-equal) but
the surviving set is **AECMOS-neutral** (all buckets ≤0.002 vs C_pb28).
`__version__` 3.22.2 → 3.22.3 (BALANCED output changed; no metric movement).

**Headline finding:** the audit's "substrate-correctness bugs that contaminate R²"
framing was **refuted**. Only two were genuine correctness fixes (reset hygiene +
window consistency), both metric-neutral. The rest are suppressor domain/tuning
*parity* changes — R² is largely decoupled from `near_psd` (R²=S²/ERLE on 92–94 %
of frames; ERLE already windowed via E1 default-ON; `capture_psd`→R² only in the
~6–8 % saturated branch). A recurring lesson: **AECMOS penalises spectral *reshape*
even when real residual-echo energy is equal or lower** — only per-case energy
audio-proof separated artifact from real regression.

**KEPT (shipped into BALANCED, AECMOS-neutral):**
- **P0.1 — coherence-gate EMA reset.** `_coh_erle_*` / `_coh_gamma2_for_floor` /
  `_coh_xy_*` were created only in `__init__` and never cleared on reset, so a
  mid-stream path-change / delay reset / cross-case instance reuse kept a stale
  Γ²(Ŷ,Y) gating the (default-ON) ERLE coherence gate. Extracted
  `_reset_coherence_state()`; `__init__` initialises and `_reset_aec3_post`
  clears through it (one funnel). 800-case: 1/800 changed (a DT_movement reset
  case), Δ≈0, −103 dB localized to the reset point; `_ours_nores` byte-identical.
  Latent in the bench (fresh `AEC` per case) but real in production streaming.
- **P0.4 — analysis window canonical sqrt-Hann.** Analysis was
  `sqrt(np.hanning(N))` (denom N−1) while synthesis is canonical periodic
  sqrt-Hann (denom N); the mismatch left **0.248 % OLA gain drift** at frame
  edges. Aligning analysis to the canonical form gives **true perfect
  reconstruction** (max|OLA-gain−1| 4e-16). 800-case AECMOS-neutral (0 cases
  |Δ|>0.1; worst case energy-neutral); `_ours_nores` byte-identical (the linear
  residual uses the rectangular `error_spec`, not the windowed path).
- **P0.5 — `erle_e2y2_gate_*` carried across reset.** The `_reset_aec3_post`
  `AecStateConfig` rebuild omitted the two gate params the `__init__` config
  sets; default-OFF so production is byte-equal, but an env-enabled gate was
  silently dropped after any mid-stream reset. Now preserved.

**GATED (reverted; documented, not shipped):**
- **P0.2a — windowed SG-nearend + CNG Y².** AEC3 windows Y² everywhere, but
  feeding the windowed nearend into the per-bin Wiener gain (against a rect-domain
  R²) **reshapes** the residual spectrum: AECMOS FS echo −0.04 (breaks
  FS_movement>3.5) while **159/167 FS regressors are quieter-or-equal in real
  energy** (mean −0.21 dB). No real benefit, ship-bar cost → gated.
- **P0.2b — CNG source `usable ? E² : Y²`** (AEC3 echo_remover.cc:452/482, the
  unclamped selected nearend, not raw Y²). Lowers the comfort-noise floor on
  usable frames → **genuine DT deg +0.009** (101>49 improvers; less near-end
  masking) but FS echo −0.045 (a CN-texture reshape artifact; real FS energy
  equal) breaks FS_movement. Gated, but kept **documented in-code as a CN-floor
  DT-deg lever** for the frontier phase (3-line change to revive).
- **P0.3 — C′ Γ²(Ŷ,Y) selected/windowed** (vs refined/rect). The only candidate
  that was a **real** regression, not an artifact: 37/64 FS regressors genuinely
  **louder** (+0.25 dB more residual echo), worst −0.36, zero benefit → gated.
- **P0.2c — windowed `capture_psd`** (→ RES R² / AecState). Dropped: structurally
  inert on the FS path (R²=S²/ERLE; ERLE already windowed; capture_psd→R² only in
  the ~6–8 % saturated branch).

Full evidence + the FS-regression diagnostic workflow: docs/v3_22.md §8.

## [Unreleased] — code hygiene (byte-equal, no algorithm change)

Behaviour-neutral cleanup on top of 3.22.2. `__version__` stays **3.22.2**
(no production behaviour change). Each commit byte-equal-verified
(`_ours` + `_ours_nores`, incl. movement bucket) before landing.

- **orchestrator dedup** (Track A): extracted `_build_sg_kwargs()` so the
  `SuppressionGain` ctor — previously copied verbatim across `__init__` and
  `_reset_aec3_post` (the two blocks were identical except the
  `SuppressorConfig`) — is built in one place (~140 duplicated lines → 2 call
  sites). Collapsed the always-true `erle_coh_gate_enabled or True` Γ²(Ŷ,Y) EMA
  guard to its real consumer set (`erle_coh_gate_enabled or
  coh_gain_floor_enabled`); the accumulators feed nothing else, so the EMA is
  skipped when neither consumer is on. 14/14 byte-identical.

- **dead-flag retire** (Track B): removed 6 closed/dud default-OFF flags + their
  gated code, plus 2 inert/dead items. Production stays byte-equal (all
  default-OFF; the OFF/default code path was preserved verbatim in every case)
  — 14/14 byte-identical. Removed flags + footprint:
  - `emura_r2_enabled` / `_alpha` / `_blend` — the cross-PSD R² EMA block +
    the `r2_emura`/`emura_r2_blend` blend branch in `ResidualEchoEstimator`.
  - `subband_wallclock_smoothing` / `fullband_wallclock_smoothing` — the per-hop
    ERLE-alpha rescaling branches in `subband_erle` / `fullband_erle` (kept the
    AEC3 per-block default alphas); also dropped the now-unused `sample_rate`
    plumbing into those estimators.
  - `use_wallclock_low_noise_render_iir` — the `_LowNoiseRenderDetector` IIR
    rescaling branch in `suppression_gain` (kept the literal 0.9 decay).
  - `erle_startup_follows_convergence` (W1') — the convergence-following startup
    branch in `erle_estimator` (kept the fixed-200-hop AEC3-strict gate).
  - `enable_lf_filter_failure_r2_injection` + 5 companions (lfinj) — the
    post-residual LF R²-injection block.
  - `filter_misadjustment_scale_p` — the `scale_p` P-scaling branch in `PBFDKF`
    (verified no caller passes True; the orchestrator `isinstance` if/else
    collapsed to one `scale_filter` call).
  Inert/dead: the two `_per_band_erl[:] = 0.1` resets (array is only ever
  `[0.1,0.1,0.1]`; init + 3 diagnostic reads kept), and the unused
  `AecState.reverb_decay()` / `get_reverb_frequency_response()` stubs. Their
  `AEC_*` env hooks in `eval_aec_challenge.py` were removed too. AecConfig: 103
  fields.

- **per-bin near-end SPP substrate** (Track C, default-OFF): added
  `NearendSpp` (python/modules/residual/nearend_spp.py)
  — an IMCRA-style per-bin near-end speech-presence probability built on
  minima tracking of the residual-to-reference power ratio `|E|²/R²` (uses the
  reliable reference, not Ŷ/ERLE; multi-frame minima separate a near-end onset
  from a reverb-tail far-end, the wall every single-lag-coherence discriminator
  hit). Wired as a default-OFF DT frontier-mover: an optional near-gate scales
  the cohxd floor *release* by `(1 − p_ne)` so the floor is only released where
  near-end is absent. Adds 6 default-`False`/conservative `AecConfig` fields
  (`nearend_spp_*`, `cohxd_nearend_spp_gate_enabled`; now 109). All paths
  guarded by the flags → production byte-equal (14/14, all buckets). Ships the
  diagnostic harness python/spp_step0_diag.py
  (synthetic-DT audio-proof). **Verdict: NULL — the near-gate does NOT move the
  DT frontier.** Step-0 audio-proof passes only with slow time constants
  (`alpha=0.02`, `minima_subwindow=200`); with those, the 800-case A/B (cohxd
  vs near-gated cohxd at `thr=5` and `thr=11` vs C_pb28) lands both near-gate
  points *on the plain-cohxd Pareto line* (matched-deg Δ ≤ ±0.012). Root cause:
  in real double-talk near-end and echo occupy the *same bins*, so a per-bin
  near-mask cannot release echo-only suppression without also touching near-end
  — confirms the voice-on-voice bin-overlap wall (consistent with the prior
  coherence-discriminator closures). Kept as default-OFF research substrate;
  full reasoning in docs/v3_22.md.

## [3.22.2] — 2026-06-02 — BALANCED: per-bin near-end blend + far-active floor −28

**Headline**: BALANCED ships `soft_nearend_blend_per_bin=True` + lowers
`min_gain_floor_far_active_db` −22 → −28 dB. The per-bin near-end blend derives
a PER-BIN `ne_w` from per-bin ENR (echo[k]/nearend[k]) so the suppression-gain
tuning is frequency-selective: near-dominant bins keep `nearend_tuning`
(gentle), echo-dominant bins use `normal_tuning` (aggressive). This lets the
deeper −28 floor cancel more echo without the broadband DT near-end cost.

**800-case (movement-agnostic) vs the −22 baseline (A)**:
- DT echo 4.042 → **4.155** (+0.113); DT deg 2.227 → 2.183 (−0.044, in-bar).
- FS echo 3.529 → 3.549; NE deg 4.047 (flat). All four ship bars met.
- per-bin recovers ~+0.058 DT deg vs the floor-alone −28 point (frequency-
  selective near protection); verified byte-identical to the
  `AEC_SOFT_NE_PER_BIN=1 AEC_FAR_ACTIVE_FLOOR_DB=-28` A/B render.
- vs refs: DT echo now 4.155 (AEC2 4.262 / AEC3 4.538); DT deg 2.183 beats
  AEC3 (1.850), below AEC2 (2.389).

`far_active_floor_db` is the documented single-knob preset axis (weak −18 /
strong −28+); per-bin blend is the base. Full default-OFF flag-campaign
evidence (incl. the cohxd reference-coherence echo lever, deferred for its
DT-deg root cause) in docs/v3_22.md.

Repo hygiene this version: consolidated the v3.21/v3.22 docs into single
`docs/v3_21.md` / `docs/v3_22.md`; removed unused tooling
(`ab_compare.py`, `v3_21_6_2_hf_trace.py`, `oracle_linear_erle.py`,
`v3_21_800case_bench.py`, `v3_21_800case_report_from_json.py`,
`v3_21_byte_equal_check.py`); byte-equal check is now eval + `cmp` (see
CLAUDE.md).

---

## [3.22.1] — 2026-06-02 — P4: delay-acquire phantom-mislock guard (default ON)

**Headline**: ships `delay_acquire_protect_converged` into BALANCED (default
ON). Guards the Path-A first delay acquisition: when the linear filter is
already cancelling (>2.5 dB windowed ERLE) at the current alignment, a "solid"
late acquisition is rejected (it would reset the filter + apply a spurious large
shift). Verdict: docs/v3_22_1_p4_delay_protect_verdict.md.

**Root cause (audio-localised)**: the bench pre-aligns the reference (GCC-PHAT)
*and then* the in-pipeline AEC3 matched filter runs — a double-alignment. On
weak/nonlinear-echo FS+DT-movement cases the matched-filter correlation surface
is flat, so it latches a noise peak (e.g. kZogUfYc +96 ms, 9xjhi +188 ms) at
confidence 1.0, double-shifts the reference, and ERLE collapses to ~0 for the
rest of the case. Production (raw ref, matched-filter-only) does not hit this
(it locks the true 16–32 ms delay); the guard is therefore harmless in
production and recovers a bench-measurement artifact.

**800-case (vs splitcfg baseline)**: 26 cases changed, net **Σecho +2.85 /
Σdeg +0.28**; biggest wins kZogUfYc +2.06, Xv7jH2 +1.51 (FS_movement echo);
genuine DT_movement deg wins waxU01 +0.236, w0QrMw +0.193. The FS/DT echo
"casualties" (wlAXM0iD −0.36, W0zK3dv0 −0.32, JtodX −0.17) are **audio-verified
AECMOS movement-quirks** — coherence(mic,far)=0.6 echo-dominant regions where
the fix removes *echo* (not near-end); near-only frames are byte-identical. Net
positive, no genuine regression. Bucket-level effect is small (FS_movement echo
3.480→3.502) — a clean win but not an AEC3-gap lever.

**Also landed (default-OFF research substrate, byte-equal — for v3.23)**:
re-exposed `subband_wallclock_smoothing` / `fullband_wallclock_smoothing` /
`use_wallclock_low_noise_render_iir` (M_full_delay bundle isolation knobs),
`h_error_refresh_erl_floor` / `h_error_floor_override` (cold-start deadlock
probes), and the `cohxd_*` selective floor-release scaffolding. All gated on
default-False/0.0 → output unchanged with flags off (verified vs baseline to
within int16 storage rounding). New diagnostic tool `oracle_linear_erle.py`
(LS-optimal achievable-linear-ERLE ceiling, sanity-validated).

**Phase 0 re-audit conclusion (motivating context)**: the linear filter is *not*
under-converging — on the worst FS cases it reaches/beats its LS-optimal 60 ms
ceiling (median under-convergence −1.0 dB); the linear→ERLE→RES handoff is
correct (usable_linear 87–97%) and RES does not over-suppress near-end. The
genuine FS echo gap to AEC3 (~0.32) is 47% nonlinear loudspeaker distortion
(LS-optimal linear leaves it at 0.09 dB — audio-verified) + ~43% RES suppression
depth (a gain-floor Pareto trade). Our back-end is a faithful AEC3 port (AEC3 is
itself pure-DSP NLRES with no nonlinear filter). The only beat-AEC3 lever is a
Hammerstein/power-filter PSD term (v3.23 candidate).

## [3.22.0] — 2026-06-01 — v3.22: split min-gain floor (DT/NE nearend preservation)

**Headline**: ships the split min-gain floor into BALANCED (default ON), the
first production change of the v3.22 arc. Closes the DT/NE nearend
over-suppression gap while holding echo above AEC2 — the only configuration
that meets all four ship thresholds simultaneously.

**Root cause (audio-localised)**: in doubletalk the RES Wiener gain over-
suppresses near-end by up to −8 dB (the linear filter only cuts −1.4..−2.9 dB).
Mechanism: near-end inflates the error → ERLE drops → R² spikes → the AEC3
min-gain floor `min_echo_power / R²` collapses to ~0, exactly when near-end is
present. A coherence-based double-talk discriminator was proven unworkable
(single-lag X–Y coherence cannot separate reverberant FS from DT — offline
oracle + audio). The working lever is a Pareto floor on the deepest suppression.

**Mechanism**: power-domain min-gain floor split by far-end activity —
* far-active (FS/DT): −22 dB. Caps the DT gain-collapse; the only FS cost is
  deep echo suppression below ~−40 dB (already inaudible), so FS echo stays
  above AEC2.
* far-silent (pure NE): −12 dB. Lifts NE near-end at zero echo cost (no echo
  present to leak).
Routing uses a per-recording latch on instantaneous far energy
(`mean(render_block²) > 1e6`, render int16-scaled; ≈ the orchestrator's own
far-active criterion). Latches from the first far frame so FS/DT use the gentler
floor throughout (no cold-start leak); only recordings where far is never active
keep the strong floor. Config: `min_gain_split_floor_enabled` (default True),
`min_gain_floor_far_active_db` (−22), `min_gain_floor_far_silent_db` (−12),
`min_gain_far_latch_power` (1e6) in [config.py](python/modules/config.py);
applied in [suppression_gain.py](python/modules/residual/suppression_gain.py)
`_get_min_gain` / `get_gain`.

**800-case (combined FS=300 / DT=300 / NE=200, vs cp_05 default-ON stack
E1+x2+E2+D3+L1+C′)**:

| metric | cp_05 | v3.22.0 | AEC2 | AEC3 | threshold |
|---|---|---|---|---|---|
| FS echo  | 3.726 | 3.520 | 3.484 | 3.875 | > 3.5 ✓ |
| DT echo  | 4.461 | 4.042 | 4.262 | 4.538 | > 4.0 ✓ |
| DT deg   | 1.952 | 2.226 | 2.389 | 1.850 | > 2.2 ✓ |
| NE deg   | 3.906 | 4.047 | 4.098 | 3.454 | ≥ 4.0 ✓ |

Only configuration to meet all four (AEC2 3/4, AEC3 2/4). vs v3.21 baseline:
FS echo +0.09, DT deg +0.13, NE deg +0.11. Audio-validated: DT near-end
recovered +1.9..7.8 dB with no echo leak in far-silent gaps; NE +1..4 dB.

**Audit by-products** (no production change): refuted via data two "critical"
claims — PBFDKF H_error step-size collapse (H_error spans 1e-3..2.0 during
movement = correct Kalman behaviour) and a delay-histogram movement latency
(the worst FS_movement case has zero mid-case delay shifts). The one verified
unused lever is the reverb accumulator's ~2.9× steady-state under-weight,
deferred as an approach-AEC3-echo stretch (echo↑/deg↓).

---

## [Unreleased] — 2026-05-29 — v3.21 CLOSE: conversion audit + Tier-C verdict + alignment-flag inlining (byte-equal)

**Headline**: closes the v3.21 AEC3-alignment arc. A deep audit of every
hop/fft "physical-meaning conversion" flag found two shipped `default-True`
flags were mislabelled "AEC3-strict" but actually mis-derived, and two were
unvalidated gray-zone conversions. The 800-case **Tier-C validation**
(`docs/v3_21_tierc_validation_report.md`) adjudicated each via the
matched-magnitude AECMOS Pareto, then all surviving alignment flags were
inlined into their call sites and all NOSHIP substrate removed. **No
production behaviour change — byte-equal preserved** (12-case `_ours` +
`_ours_nores` md5 gate, all buckets incl. movement).

### Tier-C validation verdict (800-case, isolated per flag)

- **`active_render` correction → REVERTED.** The arithmetically-correct AEC3
  value (`100²/32768² = 9.31e-6`, mean-correct vs the shipped `5.96e-4` which
  kept AEC3's block-SUM ×64 against our per-sample mean) **regressed** FS echo
  −0.033 mean + 25 catastrophic cases (worst −1.431). Production keeps
  `5.96e-4`, **relabelled as an empirically-tuned threshold, NOT strict
  alignment**; re-tuning for our hop/fft deferred to v3.22 Arc 4.
- **`fft_density` floor scaling → KEPT (inert).** 4× vs 1× ≈ 0.000 across all
  buckets. The shipped factor is `(fft/2)/64 = 4×`; the true per-bin energy
  basis is the real frame `2×hop = 320 → (2×hop)/64 = 5×` (broadband; 25×
  tonal). The 4× is a single-constant approximation that is AECMOS-insensitive;
  derivation comment corrected, signal-adaptive floor deferred to v3.22 Arc 3.
- **`reverb_smoothing` (EMA-α 0.2→0.428) → KEPT converted.** Reverb is an
  envelope/decay tracker (preserve wall-clock τ is the aligned choice), same
  class as `dne_trigger` — not a ratio estimator like the C1-C3 ERLE EMAs.
  Inert on AECMOS (0.000); keeping it preserves byte-equal.
- **`dne_trigger` (evidence count 12→5 hops) → KEPT converted.** Reverting
  tanked DT deg (−0.065/−0.075 + 51 catastrophic, 47 of them DT) for an FS
  echo gain — a losing matched-magnitude trade. The conversion is a correct
  genuine-temporal (trigger-latency) alignment.
- **Combined "honest-alignment" config (V5) → REJECTED** (DT deg −0.062 + 49
  catastrophic) — validation prevented shipping a DT regression.
- Earlier **C1-C6 wall-clock-EMA bundle → NO-SHIP** (FS echo down / illusory DT
  lift); root cause: ratio estimators (ERLE) want preserve-count, not
  preserve-seconds.

### Release cleanup (byte-equal)

- **`config.py` 412 → 230 lines.** Removed all 16 AEC3-alignment / NOSHIP /
  temp config flags. The 9 shipped `default-True` alignment flags are now
  hard-coded into their `orchestrator.py` / `filters.py` call sites; the
  C1/C3/C4/C6 + `just_reset` (CLOSED −8 dB) + `block_energy` (dormant) + the
  temp `active_render_threshold_aec3_corrected` validation flag are deleted.
- **`filters.py` 1057 → 863 lines.** Collapsed PBFDKF `_update_weights`: the
  legacy P-denominator Kalman body (default-OFF, 131 lines) is removed and the
  10 always-True refined/shadow AEC3-parity flags (`_use_aec3_h_error`,
  partition-summed X², current-E²-refined, per-bin H_error refresh, filter
  noise gate; the five PBFDAF coarse-filter gates) are inlined. Orphaned
  kx/p53 trace scaffolds removed.
- **`orchestrator.py`**: init + reset construction paths collapsed (fft-density
  if/else, dne `_dc.replace`, gain-ratchet, reverb), `_aec3_post` flag branches
  inlined (x2-reverb-for-ERLE, subtractor-max-abs, active_render → single
  value, just_reset retired, block_energy dropped). `epc.py` C6 gate → legacy
  1e-4.
- Satellite estimators (`state/subband_erle`, `state/fullband_erle`,
  `residual/suppression_gain`, `residual/reverb_frequency_response`) retain
  their internal parameterisation, now driven by hard-coded orchestrator
  literals (config-level cleanup complete; deeper leaf-branch removal is an
  optional follow-up).

### Canonical 800-case scorecard (shipped version, this entry)

Rendered via `eval_aec_challenge` (per-case `np.random.seed(0)`, ref pre-aligned
by `estimate_delay`, max_delay 1024 ms) and scored with
`model/Run_1663915512_Stage_0.onnx` (FastAECMOS). Cross-checked against the
in-memory Tier-C `V0_prod` to within ±0.011 echo / ±0.002 deg (CNG-seed margin
only — confirms the two render paths agree). Full per-bucket worst-20:
`results/v3_21_close/result.md`.

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.491 | 4.999 |
| FS_movement | 131 | 3.351 | 5.000 |
| DT_static | 186 | 4.371 | 2.059 |
| DT_movement | 114 | 4.247 | 2.159 |
| NE | 200 | 4.998 | 3.941 |

Collapsed across movement (FS/DT each = static + movement, case-weighted from
the same `scores.json`):

| Bucket | n | echo (↑) | deg (↑) |
|---|--:|--:|--:|
| FS | 300 | 3.429 | 4.999 |
| DT | 300 | 4.324 | 2.097 |
| NE | 200 | 4.998 | 3.941 |

### Repo housekeeping (this entry)

Removed closed-arc cruft: `docs/v3_21_6_2_trace.md` (superseded 3-case trace),
`docs/v3_21_close_handoff.md` (AI-continuity, folded into project memory + this
entry), `python/v3_21_tierc_validation_bench.py` (its V1–V5 variants toggle
now-inlined flags → no longer runnable; verdict captured in the validation
report). Also pruned ~1.2 GB of stale render output (`out_python/`,
`out_v3_21_800case/`, stale `results/` subdirs, `listen_ab/`). Kept
`python/v3_21_6_2_hf_trace.py` as the painted-black HF reference tracer for v3.22
Arc 2 (per the new diagnostics-discipline directive in `docs/v3_22_plan.md`:
future diagnostics fold into modules as flag-gated instrumentation, not
standalone trace scripts). `run_one_case.py` re-verified on current code
(deterministic; differs from the 800-bench only by the bench's ref pre-align).

Evidence: `docs/v3_21_tierc_validation_report.md`,
`docs/v3_21_800case_bench_report.md`, `docs/v3_21_alignment_roadmap.md`.
v3.22 roadmap: `docs/v3_22_plan.md`.

---

## [3.21.6.4] — 2026-05-28 — physical-meaning fix: ReverbModel decay wall-clock alignment

**Headline**: v3.21.6.3 trace on user's case (568_EVB_online) reduced
reverb share 64.4 % → 51.2 % via the FullBandErleEstimator hold wire-up,
but the 6.2-7 s severe-wipe segment still showed gain_hf=0.49 unchanged
because the gap between convergence events exceeded the 400 ms hold and
quality dropped to None. Re-audit of `ReverbModel::UpdateReverb` revealed
the actual physical-meaning bug: `decay = 0.83` is the AEC3 per-block
(4 ms / 64-sample) multiplier, but our pipeline calls the same update
once per 10 ms / 160-sample hop. Applied verbatim, our wall-clock T_60
was 371 ms vs AEC3's 148 ms — **2.5× too long**. When the filter is
unconverged and `tail_response` is held stale, this inflated the steady-
state reverb mass by ~2.2× via `reverb_ss = injection / (1 - decay)`
(5.88 × at 0.83 per hop vs 2.66 × at the corrected 0.624 per hop).

### Fix

`python/modules/residual/residual_echo_estimator.py` `_reverb_decay()`:
apply wall-clock alignment to both static-config decay and adaptive
estimator output:
```python
_AEC3_BLOCK_SAMPLES = 64
if self._hop_size != _AEC3_BLOCK_SAMPLES:
    d = d ** (self._hop_size / _AEC3_BLOCK_SAMPLES)
```
At our hop=160 this is `0.83 ** 2.5 ≈ 0.624`. The ratio is computed
from `_hop_size` at runtime so the conversion auto-scales if hop_size
ever changes.

`python/modules/residual/residual_echo_estimator.py` `ReverbConfig`
docstring updated: `decay = 0.83` is now correctly noted as the AEC3
per-block constant with per-hop derivation documented at the call site.

### Verification (demo case 0I0XMl3M DT_static)

v3.21.6.3 (hold wire-up only) → v3.21.6.4 (decay wall-clock fix):
- `ENR HF median` (R²/Y²): **0.945 → 0.741** (−21.6 %)
- `ENR HF mean`:           **1.457 → 1.073** (−26.4 %)
- `R²_reverb median`:      **+58.13 dB → +52.32 dB** (−5.8 dB)
- `reverb share`:          **51.2 % → 21.7 %** (−29.5 pp)
- `hf_lim_applied`:        **30.6 % → 8.4 %**
- `gain_hf_med` post-cap:  **0.036 → 0.075** (+0.039)

### What this doesn't fix

The wall-clock-aligned decay reduces R²_reverb steady-state but FS
echo cancellation strength is also reduced proportionally (faster
decay = less tail mass at steady-state to suppress real reverb).
Trade-off needs 800-case bench to confirm cohort-level Pareto. The
shipped change is AEC3-strict physical alignment — not tuning —
so it stays default-ON without flag.

### Files

- `python/aec.py`: `__version__ = "3.21.6.4"`
- `python/modules/residual/residual_echo_estimator.py`: `_reverb_decay()`
  wall-clock alignment + `ReverbConfig` docstring
- `CLAUDE.md`: version bump
- `CHANGELOG.md`: this entry

---

## [3.21.6.3] — 2026-05-28 — strict AEC3 wire-up: FullBandErleEstimator → ReverbFrequencyResponse

**Headline**: HF-painted-black per-frame trace on the user's internal case
(568_EVB_online) attributed the symptom to inflated R²_reverb (64.4 %
share, +56.6 dB int16²) — the reverb tail held STALE from a brief early
convergence and never refreshed during sustained NE. Root cause is a
v3.21 port wire-up gap: `ReverbFrequencyResponse.Update()` was fed a
binary `1.0 if converged else None` filter quality proxy instead of the
AEC3-strict continuous estimate from `FullBandErleEstimator`. AEC3 keeps
quality alive for ~400 ms past per-frame convergence via the hold
counter, extending the reverb refresh window so the tail tracks the
current filter state instead of freezing.

### Fix

`python/modules/state/aec_state.py`: add
`get_inst_linear_quality_estimate()` accessor sourcing per-frame quality
from `ErleEstimator._fullband.get_inst_linear_quality_estimate()`. AEC3
strict path: `aec_state.cc:286-289` →
`reverb_model_estimator.cc:58-66` → `reverb_frequency_response.cc:88`.

`python/modules/orchestrator.py` (`_aec3_post`): replace
```python
_converged_for_reverb = bool(_aec3_converged and _filter_converged_enough)
_filter_q = 1.0 if _converged_for_reverb else None
```
with
```python
_filter_q = self._aec3_state.get_inst_linear_quality_estimate()
```

`python/v3_21_6_2_hf_trace.py`: expose continuous `filter_quality` in the
per-frame snapshot + Flag-fractions block so the 400 ms hold can be
verified directly.

### Verification (single-case smoke, 0I0XMl3M DT_static)

- `filter_quality` alive **13.6 %** of frames vs `filter_converged` only
  **2.0 %** — hold extends live-window 6.8×, exactly the AEC3 behaviour.
- `R²_reverb (HF)` at SYMPTOM: **+58.00 dB → +52.11 dB** (−5.9 dB).
- Per-band gain at 4-6 s window: LF/MF/HF **0.81 / 0.56 / 0.66 → 0.88 /
  0.63 / 0.68**.

### What this doesn't fix

Changes WHEN reverb update fires (extends 400 ms hold) but not HOW the
tail is recomputed during sustained no-convergence. On cases where the
PBFDKF never converges for > 400 ms quality goes None and the tail
freezes. If the symptom persists, escalation candidates:
- shorten reverb decay via `ep_strength.nearend_len < default_len`
  (AEC3 has both at 0.83 by default);
- explicit stale-reverb suppression gate at the `AddReverb` call site.

---

## [3.21.6.2] — 2026-05-28 — AEC3 alignment audit + 4 shipped items

**Headline**: Full audit of the `docs/v3_21_alignment_roadmap.md` "still missing"
list against the actual codebase. 80 % of the listed items turned out to be
already-aligned, no-consumer additions, or no-op in our `current=max=13`
steady state. The audit closes 4 items that genuinely needed shipping; the
remaining gaps either require an architecture redesign (Tier C #11
`convergence_seen` latch) or carry too much risk to ship inside v3.21.x
without a dedicated design lock (Tier A #2 `IsRenderTooLow` /
`use_stationarity_properties = false`). `__version__` bumped to 3.21.6.2.

### Audit findings

| Roadmap item | Status after audit |
|---|---|
| Tier A #2 EchoAudibility config defaults | Already AEC3-strict (`floor_power = 2·64`, `audibility_threshold_{lf,mf,hf} = 10.0`, `low_render_limit = 4·64`, `normal_render_limit = 64.0`). Orchestrator override `use_stationarity_properties = True` retained as load-bearing safety net (AEC3 default is `false`; flipping is a Phase-2 design item, not shipped). `IsRenderTooLow` + `non_zero_render_seen_` latch NOT shipped — would change stationarity-noise-floor update timing on every frame; too high-risk for an unbenched bundle. |
| Tier A #3 SubtractorOutputAnalyzer | Strict AEC3 surface has only 3 signals (`any_filter_converged`, `any_coarse_filter_converged`, `all_filters_diverged`). `any_filter_converged` already wired (`bridge.filter_converged`). **Shipped**: added `any_coarse_filter_converged` (relaxed predicate `e²_c < 0.3·y²` with `kConvergenceThresholdLowLevel = 20²·hop`) and `all_filters_diverged` (`min(e²_r, e²_c) > 1.5·y²` with `30²·hop` threshold) to `FilterStateBridge`. No production consumer today (TransparentMode HMM is retired); zero-impact additive surface for Phase 2 HMM port. |
| Tier A #5 FullBandErleEstimator | Already fully ported (`python/modules/state/fullband_erle.py`) and wired via `ErleEstimator.update()`. **Shipped**: removed duplicate inline `_update_erle_inst_quality` from `orchestrator.py` (dead code — the `use_aec3_erle_reverb_quality` flag was retired in v3.21.6.1 cleanup). |
| Tier B #8 `poor_coarse_filter_counter` hangover | **Shipped, behavioural change**. Two physical-meaning corrections vs the pre-audit code: (a) trigger threshold `_threshold_hops` was 5 hops = 50 ms wall-clock; AEC3 strict `< 5` blocks = 20 ms → `blocks_to_hops(5, 160, 16k) = 2 hops`; (b) hangover was 40 AEC3 blocks (16 hops, 160 ms); AEC3 strict `coarse_reset_hangover_blocks = 25` → 10 hops, 100 ms. Also dropped the non-AEC3 `shadow_mu_scale = 0.0` freeze during hangover (AEC3 keeps coarse adapting, only the refined filter disallows `leakage_diverged` — see `subtractor.cc:264-307`). 0.5× safety margin on the trigger predicate retained (the strict `e²_r < e²_c` rule was 12-case Pareto-FAIL — `docs/v3_21_poor_coarse_rescue_12case_verdict.md`). |
| Tier C #9 `use_aec3_zero_filter_on_epc` | **Shipped (documentation port)**. Added `PBFDAF.zero_filter_partitions(old, new)` mirroring AEC3 `AdaptiveFirFilter::ZeroFilter(old, new, &H_)` (`adaptive_fir_filter.cc:460-472`). Steady-state semantics: `current_size = max_size = 13` → call is a no-op. Existing call sites (all `zero_filter=False`) unchanged. Documented `W.fill(0)` branch as PBFDKF-specific divergence not parity. |
| Tier C #10 `use_aec3_epc_classification` | No-op (depended on #9 redesign). |
| Tier A #1 DominantNearendDetector | (rescinded 2026-05-28) — already ported inline at `suppression_gain.py:311`. |
| Tier A #4 TransparentMode HMM | Deferred to Phase 2 with `SubtractorOutputAnalyzer` (consumer of new bridge signals). |
| Tier A #6 ScaleFilter (exact) | Out of scope for this audit — retiring `FilterMisadjustmentEstimator` requires its own ablation arc. |
| Tier B #7 4-mode HF tuning | Out of scope — needs separate per-band re-tune cycle. |
| Tier C #11 `convergence_seen` latch redesign | Architecture work; not in this audit. |

### Byte-equal verification (single-case smoke)

Cohort case `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` (~42 s DT_static).
- Tier A #5 dedup + Tier A #3 additive + Tier C #9 doc port (alone): md5 `b898fc57f094db6da10891b0f606240a` — **byte-equal vs v3.21.6.1 baseline**.
- All four shipped items: md5 changes (Tier B #8 hangover is the only behavioural delta; expected).

### Physical-meaning alignment (user directive 2026-05-28)

All AEC3 wall-clock conversions audited under hop=160 / block=320 / fft=512:
- `coarse_reset_hangover_blocks = 25` → 100 ms → `blocks_to_hops(25, 160, 16k) = 10 hops`.
- `poor_coarse_filter_counter < 5` blocks → 20 ms → `blocks_to_hops(5, 160, 16k) = 2 hops`.
- `kBlocksToHoldErle = 100` → 400 ms → `int(0.4 × HOPS_PER_SECOND) = 40 hops`.
- y² thresholds (`50² / 20² / 30²` × kBlockSize) → scaled by `hop/block = 160/64 = 2.5×` for our hop sums.

### Files

- `python/aec.py`: `__version__ = "3.21.6.2"`
- `python/modules/filter/filter_state_bridge.py`: `FilterStateBridge` adds `any_coarse_filter_converged` + `all_filters_diverged`; `build_filter_state_bridge()` accepts both.
- `python/modules/orchestrator.py`: computes relaxed coarse / strict diverged signals next to the existing strict-converged block; threads into bridge; removes inline R0.4 dead code; corrects poor_coarse hangover physical-meaning + drops shadow-freeze divergence.
- `python/modules/state/aec_state.py`: TransparentMode branch sources `all_filters_diverged` from bridge (falls back to legacy heuristic for back-compat).
- `python/modules/filters.py`: adds `PBFDAF.zero_filter_partitions(old, new)`; documents `W.fill(0)` ablation surface.
- `CLAUDE.md`: version bumped.
- `docs/v3_21_alignment_roadmap.md`: updated with audit verdict per item.

---

## [3.21.6.1] — 2026-05-27 — AEC3 alignment completion (nores artifact + HF deficit fixes) + release cleanup

**Headline**: Two production-blocking defects fixed on the v3.21.6 baseline, then full release cleanup (config / orchestrator / dev-trace removal). `__version__` bumped to 3.21.6.1.

### Bug 1 — nores LF artifact (linear-filter over-estimation)

PBFDKF refined-filter weight-update parity gaps caused LF (0–500 Hz) over-modelling: residual carried inverted-phase echo instead of cancelling cleanly. Five AEC3-strict fixes shipped, all now hard-coded into the call sites (no flags):

- `RefinedFilterUpdateGain` denominator uses `SpectralSum = Σₚ|X_buf[p]|²` (was: `X²_latest` of current hop only). AEC3 `refined_filter_update_gain.cc:103-107`.
- Per-bin H_error refresh from `|error_spec|²` per bin, gated `e2_refined_per_bin ≤ e2_coarse_per_bin` (was: scalar fullband leakage compare). AEC3 `cc:128-138`.
- H_error ceiling `= 2.0` in float (was: `= 100.0` legacy int16-derived). AEC3 `kHErrorCeiling`.
- Refined filter noise gate constant = `20075344` (was: `27509562` borrowed from suppression path — 1.37× too tight). AEC3 `refined.noise_gate` + `coarse.noise_gate`.
- μ denominator `n_part·E²` term uses current-block `|error_spec|²` (was: smoothed `_error_psd` α=0.95 with ~200 ms lag). AEC3 `SubtractorOutput.E²_refined`.

Matching PBFDAF shadow protection (A.1–A.5: partition-summed X² for shadow μ / noise gate / poor-excitation gate / narrowband mask / saturation gate) also hard-coded ON. AEC3 `coarse_filter_update_gain.cc:34-82`.

### Bug 2 — painted-black HF (residual chain over-suppression)

Internal DT cohort case showed > 4 kHz bins painted black (gain_hf_median < 0.1) on 24.3% of frames; reverb-to-direct R² ratio median 6.2 / max ~10⁴. Ten alignment commits brought reverb chain + back-end to AEC3 strict:

- **Reverb-render-history bug**: `ReverbModel` was fed current render PSD instead of the render block from N+1 partitions ago. Added `_reverb_render_history` deque (`maxlen=16`); read `[n_partitions + 1]` for linear mode, `[filter_delay_blocks + 1]` for nonlinear mode. AEC3 `residual_echo_estimator.cc:367-376`.
- **RFR neighbour window**: reverted from ±4-bin expansion back to AEC3-strict ±1-bin point-wise max. Wider window inflated sparse HF energy.
- **ReverbConfig defaults**: `decay = 0.83` (was 0.85), `mild_decay_scale = 1.0` (was 0.5), `use_adaptive_decay = False` (was True). AEC3 `ep_strength.default_len = 0.83`.
- **Audibility band boundaries**: `lf_band_end_hz = 375.0`, `mf_band_end_hz = 875.0` (was: 94 / 219 Hz from legacy 65-bin port). AEC3 `WeightEchoForAudibility` at fft=128.
- **HF cap anchor**: 2000 Hz (was: 4000 Hz) with 1-bin physical width 31.25 Hz (was: 125 Hz). AEC3 `limiting_gain_freq_hz`.
- **CNG strict alignment**: AEC3 3-state N2 estimator (Y² EMA → N2 update with 50-block onset / 1.0002 slow-up / 0.9·0.1 track-down → N2_initial 1000-block transient → noise floor clamp) + `sqrt(1 − G²)` injection (no 0.4 scale) + LCG random + `sqrt(2)·sin` LUT phase. AEC3 `ComfortNoiseGenerator`.
- **OLA synthesis window**: MATLAB-canonical sqrt-Hann (denom `N`) replacing numpy `np.hanning` (denom `N − 1`).
- **Stationarity window**: auto-rescales via `blocks_to_hops(13/12, hop, sr)`. Was: literal 13/12 raw hops treating AEC3 4 ms blocks as 10 ms hops (130 ms vs intended 52 ms).
- **Delay estimator**: `detect_pre_echo = True` (AEC3 strict per `echo_canceller3_config.h:73`).
- **FilterPlateauDetector**: moved to opt-in flag (Python-only safety net, no AEC3 equivalent). Default off for strict alignment.

### Cohort DT case progression (internal test)

| Metric | v3.21.6 baseline | + reverb fix | + full alignment |
|---|---:|---:|---:|
| Painted-black HF (`gain_hf_median < 0.1`) | 24.3% | 15.1% | **14.0%** |
| Healthy frames | 60% | 75% | **76.3%** |
| Rev/direct R² ratio (median) | 6.2 | 0.13 | **0.081** |
| Rev/direct R² ratio (p95) | ~10⁴ | 170 | 170 |

Residual 14% painted-black is in the AEC3-inherent regime: `bin/aec3_cli` itself fails the same threshold on the cohort DT_movement cases. Further improvement requires beyond-AEC3 work (deferred to v3.22).

### Release cleanup (12 commits, audio-zero)

After alignment shipped, dev-time substrate was removed. **Byte-equal vs `cd73f4e` (alignment HEAD) verified at every commit (10/10 MD5 PASS).**

| Area | Before | After | Reduction |
|---|---:|---:|---:|
| `config.py` | 1547 | 160 | −90% |
| `orchestrator.py` | 5487 | 3632 | −34% |
| `filters.py` | 1062 | 1020 | −4% |
| `eval_aec_challenge.py` | 1169 | 779 | −33% |

- **Config**: linear-filter parametric tuning (`mu`, `kalman_*`, `shadow_*`, `filter_misadjustment_*`) kept as INTERNAL for preset backward-compat; AEC3 alignment flags hard-coded into call sites; NOSHIP / ablation / trace flags removed entirely. Customer surface is now residual-side only (`enable_res`, `enable_cng`, `comfort_noise_floor_dbfs`, HPF, saturation, delay-est).
- **Orchestrator**: removed Q1 transient leakage, F-E1/E3/E5/DelayTrack helpers, DT advisory + diverged-reset chains, kalman_q_per_band + Arc M tilts, all per-frame trace dicts (`_hf_chain_trace`, `_uro_trace` etc.), `__getattr__` shim that the first cleanup agent inserted.
- **Modules deleted**: `state/transparent_mode.py`, `state/signal_dependent_erle.py`, `nlp.py` (SubtractiveNLP closed CANNOT-SHIP per its own docstring).
- **Eval driver**: stripped ~50 dev env-var hooks (`AEC_PLAN_A_*`, `AEC_F_E*`, `AEC_SHADOW_*` etc.). Surviving env vars: `AEC_GAIN_TYPE`, `AEC_MODE`, `AEC_MAX_DELAY_MS`.
- **Docs**: 110 v3.21/v3.22 trace + verdict files removed. Kept: `aec_methods.md`, `aec_algorithm_guide.html`, `architecture_v3_10_5_vs_v3_21_vs_aec3.html`, `c_user_and_integration_guide.md`, `dtd_design.md`, `pbfdkf_shadow_intro.md`, `linear_filter_evolution.svg`, `aec3_extracts/`.
- **Misc**: renamed `aec_filter_evolution.svg` → `linear_filter_evolution.svg` (clearer scope vs the residual / post-filter SVGs). Removed `python/check_byte_equal.py` (anchored to deleted baseline JSON).

### AEC3 alignment audit — what shipped, what didn't, why

**22 / 22 should-be-ON flags shipped (hard-coded into call sites)**: `use_aec3_filter_misadjustment_parity`, `use_per_bin_h_error_refresh`, `use_aec3_h_error_ceil`, `use_partition_summed_x2_for_h_error_gain`, `use_aec3_filter_noise_gate_power`, `use_aec3_residual_noise_gate`, `use_aec3_echo_gen_power_window`, `use_aec3_handle_echo_path_change`, `use_full_delay_change_chain`, `use_current_e2_refined_in_h_error_denominator`, `use_refined_output_selection_for_linear_path`, `form_linear_filter_crossfade_enabled`, `use_partition_summed_x2_for_shadow_mu`, `use_aec3_noise_gate_for_shadow`, `use_poor_excitation_gate_for_shadow`, `use_narrowband_mask_for_shadow`, `use_saturation_gate_for_shadow`, `filter_misadjustment_enabled`, `filter_analyzer_enabled`, `e2_y2_clamp_enabled`, `aec3_post_stationarity_zero_enabled`, `shadow_class_nlms`.

**6 NOT shipped** (each with documented reason for future re-evaluation):

| AEC3 flag | Reason NOT shipped | Next-step gate |
|---|---|---|
| `use_aec3_erle_reverb_quality` | Requires `FullBandErleEstimator` port (continuous quality 0→1 for reverb model). Substrate gap. | Port FullBandErleEstimator first, then re-test. |
| `use_aec3_zero_filter_on_epc` | `W.fill(0)` on EPC destructive on cohort tail (~7/800 cases); `PathChangeRegimeHandler` is load-bearing replacement. | Needs PBFDKF-aware EPC handler redesign before any flip. |
| `use_aec3_epc_classification` | Depends on `use_aec3_zero_filter_on_epc`; both fail Pareto together. | Re-evaluate jointly after zero_filter dependency clears. |
| `use_coarse_e2_time_domain_parity` | Confirmed 24/24 byte-equal no-op (threshold-bound dormant). | Document-only; not a real divergence. |
| `use_aec3_poor_coarse_rescue_copy` | 12-case Pareto FAIL: xFk7 +0.195 / MYrVxVEM −0.431 / qNvSMyUS −0.216 / 9xjhi −0.112. Conditional gating belongs in v3.22 beyond-AEC3, not strict alignment. | Defer to v3.22. |
| `use_linear_filter_output_selection_for_final_output` | `usable_linear` latch contaminates Y-vs-E selection on cohort tail. | Needs `convergence_seen` latch redesign first. |

### Commit ledger

**Alignment phase** (10 commits — `219ee2a..cd73f4e` — audio intentionally changed):

| Hash | Subject |
|---|---|
| `219ee2a` | HF cap anchor 4000 Hz → 2000 Hz (root cause of DT HF black block) |
| `795549b` | HF cap anchor width 125 Hz → 31.25 Hz (strict 1-bin semantic) |
| `8aafe61` | reverb_freq_response fft-resolution-aware smoothing (reverted in `a44dd6d`) |
| `9d77c3b` | audibility band boundaries 94/219 Hz → 375/875 Hz |
| `1650153` | strict CNG + OLA window port (back-end alignment) |
| `e7db416` | CNG constants auto-rescale by hop/sr (wall-clock parity) |
| `1209fa0` | HF paint-black attribution fields in `_hf_chain_trace` (diag only) |
| `a44dd6d` | reverb chain alignment (paint-black HF root cause — 3 reverb bugs) |
| `d311b24` | `nearend_average_blocks` wall-clock rescale |
| `cd73f4e` | 3 alignment fixes from Bundle/Plateau/Stat/Delay audit |

**Cleanup phase** (13 commits — `1b11933..14b3327` — audio zero-impact, byte-equal verified each step):

| Hash | Subject |
|---|---|
| `1b11933` | remove v3.21/v3.22 dev trace docs + scripts (140 files, −55 416 lines) |
| `55872db` | collapse AecConfig dev-time substrate flags (~85 fields) |
| `11d7525` | remove orchestrator dead flag branches |
| `88e84fa` | hard-code shipped flags in filters.py + final orchestrator pass |
| `5cd93bc` | remove SubtractiveNLP module (closed CANNOT-SHIP substrate) |
| `2b59d05` | remove SubbandNearendDetector linear-filter mode |
| `4e945fd` | remove plateau detector + DT advisory dead branches |
| `092da0a` | delete F-E1/E3/E5/DelayTrack dead state + helpers |
| `8139418` | remove AecConfig `__getattr__` legacy-flag shim |
| `587955b` | strip version-history archaeology comments |
| `53759e1` | drop write-only diagnostic state + dead trace dicts |
| `0035dd8` | strip dev-time env-var hooks in eval_aec_challenge + rename evolution svg |
| `14b3327` | v3.21.6.1 release: docs + version bump |

### Verification

- **Byte-equal**: 10/10 MD5 match `cd73f4e` baseline on 5-case sample (NE / FS_static / FS_movement / DT_static / DT_movement) after the full 13-commit cleanup phase.
- **Sanity tests**: 25/25 pass (`test_p52_regime.py` 18 + `test_aec_reset.py` 7).
- **800-case AECMOS** (preset `BALANCED`, fl=832, `--cng`, 4 workers; raw data in `results/v3_21_6_1_release/scores.json`):

  | Bucket | n | echo (↑) | deg (↑) |
  |---|---:|---:|---:|
  | FS_static | 169 | 3.917 | 4.999 |
  | FS_movement | 131 | 3.814 | 4.999 |
  | DT_static | 186 | 4.543 | 1.916 |
  | DT_movement | 114 | 4.437 | 1.942 |
  | NE | 200 | 4.998 | 3.750 |

  vs v3.21.0 baseline (architecture doc reference, `3aadd2d`): FS echo +0.19 / DT echo +0.27 / DT deg −0.46 / NE deg −0.30. Pareto trade — stronger echo cancellation, weaker preservation. Per `feedback_aecmos_pareto_comparison.md`: deg-only comparison invalid; matched-magnitude or full-Pareto evaluation required before any reference comparison. Internal listen-test verification pending.

### Branch / remote operations

- `cleanup/v3_21_6_release` work branch deleted locally after merge into target.
- `debug/v3_21_6_nores_artifact` renamed → `v3_21_release` (force-pushed to origin, old remote ref deleted).
- `feature/v3_23` deleted locally (v3.23 cycle closed, no production output).
- Remote `refs/heads` final state: `main` (unchanged), `v3.16` (legacy), `v3_21_release` (HEAD = `14b3327`).

---

## [3.21.6] — 2026-05-21 — AEC3 Parity Completion (Sprint P1 ships; Sprints P2 / P4 closed intentionally-incompatible; Sprint P3 byte-equal structural)

**Headline**: 1 production change shipped — **Sprint P1 AEC3 FilterAnalyzer port** (single-channel verbatim port of [`docs/aec3_extracts/src/aec3/filter_analyzer.cc`](docs/aec3_extracts/src/aec3/filter_analyzer.cc), `~250 LOC`, owned by `AecState`; default-True). The port produces a non-zero direct-path delay scalar that AEC3's reverb-tail update consumes — indirectly closes v3.21.5 Sprint C's reverb-tail blocker. Cumulative 800-case bench Pareto-positive vs v3.21.5: FS_static **+0.059** / FS_movement **+0.036** / DT buckets within ±0.01 / NE flat.

Sprints P2 / P4 closed as **intentionally-incompatible with our PBFDKF architecture** — both AEC3 parity items are permanently retired (TransparentMode + AEC3-default-off stationarity); any v3.22+ revisit must be labelled as PBFDKF-specific divergence, NOT AEC3 parity restoration. Sprint P3 ships byte-equal structural parity (canonical control surface for `use_stationarity_properties` now lives at `SuppressorConfig.echo_audibility`, with top-level `aec3_post_stationarity_zero_enabled` retained as deprecated alias).

Cycle close: `docs/v3_21_6_cycle_close.md`. v3.22 entry gate (AEC3 parity baseline locked): **MET** — every Bucket-1 item has a closed verdict.

### Sprint P1 — FilterAnalyzer port SHIPPED (Pareto-positive)

AEC3 [`filter_analyzer.cc`](docs/aec3_extracts/src/aec3/filter_analyzer.cc) produces `FilterDelaysBlocks()` (per-channel direct-path delay) + `ConsistentFilterDetector` (peak-stability gate for `any_filter_consistent`). Pre-v3.21.6, our [`python/modules/state/filter_delay.py:57-60`](python/modules/state/filter_delay.py#L57) received `analyzer_filter_delay_estimates_blocks=None` always (FilterAnalyzer was a v3.18 audit-only stub at `python/modules/filter_analyzer.py`, since deleted) → `min_direct_path_filter_delay()` returned 0 → reverb-tail update never fired on cohort-tail cases. This was the root cause v3.21.5 Sprint C diagnosed but couldn't fix in the v3.21.5 narrow scope.

P1 ships a full single-channel port (block units translated AEC3 `kBlockSize=64` 4ms → our `HOP_SAMPLES=160` 10ms; convergence hold 5s = 500 hops; consistency hold 1.5s = 150 hops). The new `state/filter_analyzer.py` (~250 LOC) covers `ConsistentFilterDetector` + 3-tap 600Hz HPF + region-sweep peak finder + state machine, verbatim against the AEC3 source. `AecState` owns the analyzer; `PBFDAF.get_time_domain_filter()` (new ~10 LOC IFFT concat helper) feeds the time-domain impulse response per hop. Reverb-update `_delay_blocks` switches from legacy `_current_delay // hop_size` to `aec_state.min_direct_path_filter_delay()`. The v3.18 Phase C.A audit-only stub (incompatible API) is deleted.

Verified default-OFF byte-equal preserved (25/25 PASS vs v3.21.5 anchor) before flipping default True. 800-case bench Pareto-positive on FS without DT damage. Verdict: `docs/v3_21_6_p1_filter_analyzer_verdict.md`.

Known limitation (not blocking ship): `fa_consistent=0%` on the LN18k5r8 cohort case — PBFDKF's Kalman peak position is noisier than AEC3's NLMS-stable envelope, so the 1.5s peak-stability detector rarely fires. Effect: `UpdateFilterGain` falls back to running-max path; `TransparentMode.any_filter_consistent` stays False (irrelevant — TM disabled by P2). Does not affect `filter_delays_blocks()` output (P1's primary deliverable). Documented as the PBFDKF-vs-AEC3 architectural-incompatibility note that motivated P2's parity closure.

### Sprint P2 — TransparentMode audit CLOSED intentionally-incompatible

4 mismatch findings vs AEC3 source ([`transparent_mode.cc`](docs/aec3_extracts/src/aec3/transparent_mode.cc) Legacy variant + [`aec_state.cc:189-325`](docs/aec3_extracts/src/aec3/aec_state.cc#L189) Update flow):

- (A) `enable_transparent_mode=False` hard-coded in orchestrator with stale rationale citing the v3.20 legacy 10-frame ERLE latch (already retired in v3.21 by the per-frame e²<0.5·y² gate in `_aec3_post`)
- (B) 3 block-unit constants in `transparent_mode.py` left at AEC3 4ms-block values with a misleading "blocks not hops -> stays N" comment; actually wall-clock durations
- (C) `any_coarse_filter_converged` not threaded into `TransparentMode.update` (Legacy ignores; HMM variant not ported)
- (D) `all_filters_diverged` derived from `bridge.divergence_indicator > 1.0` proxy (vs AEC3 SubtractorOutputAnalyzer)

P2.0 cohort 3-case trace (LN18k5r8 / s90M7MOT / 9xjhi + 2 others) with `AEC_TRANSPARENT_MODE=1` showed LN18k5r8 fires TM 23.1% @ fa_consistent=0% — the exact PBFDKF-vs-AEC3 cohort-tail false-activation pattern P1's verdict had already documented for FilterAnalyzer. The historical investigation protocol accepted this cohort evidence to close as intentionally-incompatible without a full 800-case bench.

Per-mismatch verdicts:
- A → intentionally-incompatible (production stays `transparent_mode_enabled=False`)
- B → **fixed dormant** (`_SANE_FILTER_DELAY_BLOCKS=5` → `_SANE_FILTER_DELAY_HOPS=2`; `_DIVERGED_SEQ_BOUND=60` → `_DIVERGED_SEQ_BOUND_HOPS=24`; `_NUM_CONVERGED_BLOCKS_HIGH=50` → `_NUM_CONVERGED_BLOCKS_HIGH_HOPS=20`; parity correctness with zero current behavior impact)
- C → aligned no-op
- D → aligned via different source signal

Parity substrate (config flag, `AEC_TRANSPARENT_MODE` env hook, 3 corrected constants, trace field) shipped dormant as v3.22 G.2 substrate. **v3.22 G.2 must be PBFDKF-specific divergence (e.g., Kalman-state-derived "no echo path" criterion / delete subsystem / keep dormant) — must NOT claim AEC3 parity restoration.** Verdict: `docs/v3_21_6_p2_transparent_mode_audit_verdict.md`. Discipline rule recorded as feedback memory.

### Sprint P3 — EchoAudibilityConfig structural wiring SHIPPED (byte-equal)

Promoted existing `EchoAudibilityConfig` dataclass (already had AEC3 audibility thresholds + render-floor knobs + use_stationarity_properties + band boundaries) from `SuppressionGain.__init__`-internal local instance to `SuppressorConfig.echo_audibility` field. Orchestrator's stationarity zeroing block at `_aec3_post:3500-3520` (two consumer sites) now reads canonical `self._aec3_sg_config.echo_audibility.use_stationarity_properties`; top-level `AecConfig.aec3_post_stationarity_zero_enabled` retained as DEPRECATED ALIAS propagated via `dataclasses.replace` at orchestrator init (the existing dataclass is `frozen=True`).

Mid-implementation pitfall caught immediately: an initial duplicate `EchoAudibilityConfig` definition clobbered the existing rich one's fields; AttributeError on smoke-render flagged it → reverted to use the existing dataclass. Single-case md5 identical pre/post at default-True; env override `AEC_STATIONARITY_ZERO=0` still produces differing output (alias path verified working).

Removal of the deprecated alias is deferred to v3.22 Sprint I cleanup (after P4 verdict). Per P4 outcome (below), the recommendation is to **keep** the alias as a research toggle indefinitely. Verdict: `docs/v3_21_6_p3_echo_audibility_wiring_verdict.md`.

### Sprint P4 — Stationarity default-off re-test CLOSED intentionally-incompatible

Re-tested the v3.21.5 Sprint B hypothesis: that P1 FilterAnalyzer + P2 TransparentMode audit + P3 EchoAudibilityConfig wiring may have rescued `_DominantNearendDetector.is_nearend_state()` firing under stationary-far conditions, making AEC3-default-off (`use_stationarity_properties=False`) safe to flip on our cohort.

P4.0 cohort 3-case re-trace (Sprint B's worst 3: WcK0OrF / wVYSGV / xQEUtY2) on post-P1+P2+P3 baseline with user-set strict 3-criterion gate. **All 3 criteria FAIL**:

| Criterion | Result |
|---|---|
| Catastrophic gain drops (Δgain_100 < -0.3) disappear | ✗ 233 / 235 / 424 frames per case (Sprint B baseline was ~280 on xQEUtY2) |
| `is_nearend_state` rate notably improves | ✗ **ΔNE = +0.0 on all 3 cases** — xQEUtY2 stays at 7.0% vs Sprint B's 7.2% baseline |
| Formant damage (1-4 kHz Δ dB) eliminated | ✗ HF Δ -0.94 dB on xQEUtY2 (still audible) |

Root cause: P1 / P2 / P3 paths don't feed into the `_DominantNearendDetector` ENR/SNR decision. The Sprint B safety-net evidence (stationarity zeroing compensates for the incomplete detector port — AEC3 has ScaleFilter / FilterMisadjustmentEstimator companions we don't port) holds on the post-P1+P2+P3 baseline. Hypothesis falsified by direct trace evidence.

Per user directive ("若 P4.0 fail，直接 close P4 intentionally-incompatible，v3.21.6 保留 zeroing default True"), **no 800-case bench run**. Production stays `aec3_post_stationarity_zero_enabled = True` permanently for our PBFDKF + RES port. AEC3-default-off `use_stationarity_properties=False` retired as Bucket-3 closed-DSP decision. Any v3.22+ revisit (e.g., port the missing AEC3 ScaleFilter / FilterMisadjustmentEstimator companions, or replace the detector with a PBFDKF-Kalman-aware NE detector) must be labelled as PBFDKF-specific divergence, NOT AEC3 parity restoration. Verdict: `docs/v3_21_6_p4_stationarity_retest_verdict.md`.

### Cumulative bench

Standard 800-case render (j9, no env overrides — exercise defaults) against `docs/bench/v3_21_5_baseline/scores.json`:

| Bucket | n | echo Δ | deg Δ | verdict |
|---|---:|---:|---:|---|
| FS_static | 169 | **+0.059** | -0.000 | ok |
| FS_movement | 131 | **+0.036** | -0.000 | ok |
| DT_static | 186 | +0.029 | -0.009 | ok |
| DT_movement | 114 | +0.016 | +0.008 | ok |
| NE | 200 | +0.000 | -0.001 | ok |

Per-case distribution (Δ < -0.05 / Δ > +0.05 threshold): FS_static 17r/68i echo; FS_movement 15r/39i echo; DT_static 11r/48i echo + 51r/56i deg (balanced); DT_movement 7r/23i echo + 17r/29i deg (net positive); NE flat. Identical to per-sprint P1.3 results (P2/P3/P4 don't change algorithmic ship state).

### vs AEC2 / AEC3 reference scores (post-v3.21.6)

Per `docs/aec_methods.md`: v3.21.5 already beat AEC2 by +1.12 FS and beat AEC3 by +0.52 DT_deg / +0.60 NE. v3.21.6 widens the AEC2 FS_static advantage to ~+1.18 (+0.059 on top of +1.12); DT_deg / NE advantage over AEC3 unchanged. AEC3 parity is **structurally complete**: every Bucket-1 item has a closed verdict (shipped / no-leverage / intentionally-incompatible). The two intentionally-incompatible closures (TransparentMode + stationarity-default-off) document permanent PBFDKF-architecture-specific deviations that v3.22+ would only re-open as labeled divergence designs.

### Discipline rules established

- `feedback_no_parity_claim_for_divergence` (local memory note): when AEC3 parity closes as intentionally-incompatible, successor design in a later cycle (e.g., v3.22 G.2 after v3.21.6 P2) must be labelled as intentional divergence with PBFDKF-specific rationale — must NOT claim AEC3 parity restoration. Mirror-image of the Round-7 "no parity smuggling into v3.22" anti-pattern.

---

## [3.21.5] — 2026-05-21 — Safe AEC3 Parity (Sprint A ships; Sprints B / C / C2 closed)

**Headline**: 1 production change shipped — Sprint A E2 = min(E2, Y2)
clamp (AEC3 `echo_remover.cc:495-501` port-fidelity fix; default-True).
Sprints B / C / C2 all closed without shipping. Cumulative bench
(A only, since B+C+C2 closed) Pareto-positive vs v3.21.4: FS_static
+0.033 / FS_movement +0.035 dB. DT deg AECMOS-sensitive but not audible
per user spectrogram check.

Historical three-cycle investigation: v3.21.5 safe parity / v3.21.6 parity
completion / v3.22 intentional divergence. Triage policy and Bucket-1 closure
status for each item are recorded below.

### Sprint A — E2 = min(E2, Y2) clamp SHIPPED (Pareto-positive)

AEC3 [`echo_remover.cc:495-501`](docs/aec3_extracts/src/aec3/echo_remover.cc#L495)
specifies `E2 = min(E2, Y2)` when `UsableLinearEstimate()` is True
(bounds residual PSD by mic PSD). Our pre-v3.21.5 [`orchestrator.py:3479-3481`](python/modules/orchestrator.py#L3479)
cited the AEC3 contract in a comment but the clamp itself was absent.
When `error_psd > near_psd` on some bins, unclamped `nearend_pwr` was
inflated → `DominantNearendDetector` ENR (= echo / nearend) biased low
→ detector mis-triggered nearend → `SuppressionGain` used conservative
`nearend_tuning` → echo leaked through HF bands.

Verdict: docs/v3_21_5_phase1_a_e2_y2_clamp_verdict.md.
Action: `e2_y2_clamp_enabled: bool = True` (default-True) in
[`config.py:222`](python/modules/config.py#L222); flag retained for A/B (set
False for byte-equal vs v3.21.4).

### Sprint B — Stationarity AEC3-default-off CLOSED REJECTED (load-bearing safety net)

AEC3 [`echo_audibility.h:40-51`](docs/aec3_extracts/src/aec3/echo_audibility.h#L40)
+ [`residual_echo_estimator.cc:303-313`](docs/aec3_extracts/src/aec3/residual_echo_estimator.cc#L303)
gates stationarity-driven R² scaling by `EchoCanceller3Config::EchoAudibility.use_stationarity_properties`
(AEC3 default = False). Our pre-v3.21.5 [`orchestrator.py:3471`](python/modules/orchestrator.py#L3471)
unconditionally zeroed R² on stationary bins. Sprint B introduced
`aec3_post_stationarity_zero_enabled` flag with default = False
(AEC3-default-off) attempting port fidelity restoration.

800-case bench: bucket means Pareto-acceptable (FS +0.032/+0.048) BUT
**62 DT cohort-tail cases with Δdeg < −0.05** (>> strict halt 30) and
audio listen showed **xQEUtY2 worst formant Δ -2.12 dB F1** (6-10× worse
than Sprint A; audible-grade attenuation, not metric noise). xQEUtY2
trace deep-dive: 6 catastrophic segments totalling ~2.9 s in 40-s case
where `gain_100` drops from ~1 → ~0 when `stationary_mask_active=100%`
AND `is_nearend_state=0%` (detector mis-fires under stationary-far,
SuppressionGain uses aggressive far-tuning → NE-speech destroyed).

Root cause: the legacy stationarity zeroing is a **load-bearing safety
net** compensating for our incomplete AEC3 detector port (missing
companion `ScaleFilter` + `FilterMisadjustment` that keep
`is_nearend_state` correctly firing on stationary-far).

Verdict: docs/v3_21_5_phase1_b_stationarity_gate_verdict.md.
Action: `aec3_post_stationarity_zero_enabled: bool = True` (default-True
restored — load-bearing legacy zeroing kept). Byte-equal 25/25 vs
v3.21.4 70e7f96 verified. Re-test scheduled for **v3.21.6 Sprint P4**
after companion mechanisms ported (P1 FilterAnalyzer + P2
transparent_mode audit + P3 EchoAudibilityConfig structural wiring).

### Sprint C — Reverb AEC3 semantic audit CLOSED diagnose-only

3 early-return paths in [`reverb_frequency_response.py:61-65`](python/modules/residual/reverb_frequency_response.py#L61)
investigated via Sprint 0 trace fields. Root cause: upstream FilterAnalyzer
stub (`aec3_min_direct_path_blocks=0` always; `linear_filter_quality=None`
90.5%). Both v3.21.5 fix candidates degenerate; cannot ship in v3.21.5
narrow scope. Verdict: docs/v3_21_5_phase1_c_reverb_semantic_audit_verdict.md.
Moved to **v3.21.6 Sprint P1** (FilterAnalyzer port — parity) +
**v3.22 Sprint F** (RES-internal dead-tail fallback — divergence).

### Sprint C2 — Per-bin H_error refresh selector re-evaluation CLOSED no-leverage

AEC3 `H_error += factor * erl` refresh already exists at [`filters.py:625`](python/modules/filters.py#L625);
`use_per_bin_h_error_refresh: bool = False` flag gates a per-bin REFINED/COARSE
selector path. v3.21.4 U4.A standalone 800-case retest closed FAIL
(17% per-case regression). Sprint C2 hypothesis: Sprint A's E2 clamp
might mask the per-bin tracking damage on DT cases. C2.0 gate test
(5 FS_static worst cases, A only vs A+C2): **9xjhi Δ erle_100 = -0.01 dB**
(memory predicted +8.42 dB single-case win from 2026-05-18 tracer; no
longer reproduces on v3.21.5 baseline). Gate fail → close as no-leverage.
Verdict: docs/v3_21_5_phase1_c2_per_bin_h_error_refresh_verdict.md.
Path stays dormant research code; env hook `AEC_PER_BIN_H_ERROR_REFRESH`
retained for re-evaluation after v3.21.6 P1/P3 may again shift canonical state.

### Sprint 0 — Trace instrumentation extension (byte-equal preserving)

Extended [`orchestrator.py:3514-3564`](python/modules/orchestrator.py#L3514)
`trace_hf_chain` schema with 22 new per-frame fields covering Sprint A
E2 clamp evidence, Sprint B stationarity mask activity, Sprint C reverb
update paths, NE state staleness, and v3.21.6 / v3.22 reservation fields.
Default-OFF, byte-equal preserving (25/25 PASS at default `trace_hf_chain=False`).

### Cumulative bench (v3.21.5 = A only) vs v3.21.4 baseline

| Bucket | n | Δecho | Δdeg | direction |
|---|---:|---:|---:|---|
| FS_static | 169 | **+0.033** | -0.000 | target direction ✓ |
| FS_movement | 131 | **+0.035** | -0.000 | target direction ✓ |
| DT_static | 186 | +0.013 | -0.014 | echo↑ deg↓ small (AECMOS-only, not audible) |
| DT_movement | 114 | +0.013 | -0.019 | echo↑ deg↓ small (AECMOS-only, not audible) |
| NE | 200 | +0.000 | +0.002 | neutral |

Per-case Pareto: FS 3.7:1 improvement:regression; DT echo 2.3:1; NE
neutral. Audio listen on 5 DT worst-dreg cases (qiQL0BUP / Je6gJ7y1 /
y2ZCo1jA / hF9Lfjcn / I2bme08k) confirms formant-band Δ ≤ 0.4 dB —
AECMOS metric over-reacts to micro-artifacts that don't manifest as
audible damage. User spectrogram check: "看起來差不多".

Cumulative FS recovery vs v3.21.0 baseline (where FS regression was
−0.218 / −0.181): A recovers ~1/3 of the gap. Remaining FS gap to be
addressed in v3.21.6 (FilterAnalyzer port may revive reverb tail update
naturally) + v3.22 (intentional non-AEC3 divergence: hybrid residual /
HF cap-NE decoupling / etc.).

### vs AEC2 / AEC3 reference scores (post-v3.21.5)

Per `docs/aec_methods.md` reference table — v3.21.5 still beats AEC2 by
+1.12 FS (echo bucket) and beats AEC3 by +0.52 DT_deg / +0.60 NE.
Sprint A's small DT deg regression does not threaten the reference
advantage. AEC3 parity is now PARTIAL (E2 clamp shipped; stationarity
gate / FilterAnalyzer port / EchoAudibilityConfig wiring deferred to
v3.21.6).

---

## [3.21.4] — 2026-05-21 — Audit cycle (4 v3.21.2 carry-overs closed; structural ms-based refactor)

**Headline**: research / audit cycle. All 4 v3.21.2 carry-overs from
the original plan adjudicated; 0 production code changes shipped. One
structural refactor (ms-based time-domain config). Byte-equal at
default config vs v3.21.3.

### V4 — Time-domain unit-conversion "bugs" CLOSED NOT-A-BUG

The v3.21.2 plan flagged 3 HIGH-severity "time-domain unit-conversion
bugs" parallel to the bin-index bug fixed in v3.21.2:
`DominantNearendConfig.trigger_threshold=12`, `hold_duration=50`,
`EchoModelConfig.noise_floor_hold=50` — bare-value ports from AEC3
(4 ms blocks) into our 10 ms hops, giving 2.5× longer wall-clock than
AEC3 intended.

V4.1 (`trigger_threshold` 12 → 5 = `blocks_to_hops` canonical) tested
empirically: bench result was **both directions worse** (FS_static
echo −0.021 / FS_movement −0.027 / DT_static deg −0.027 / DT_movement
deg −0.007). Strict regression, not Pareto.

User redirect: physical-meaning analysis (not wall-clock) is the
correct yardstick. AEC3 source inspection
([dominant_nearend_detector.cc](docs/aec3_extracts/src/aec3/dominant_nearend_detector.cc) +
[residual_echo_estimator.cc:340-358](docs/aec3_extracts/src/aec3/residual_echo_estimator.cc#L340))
confirmed each counter measures a different physical quantity:

- `trigger_threshold`: statistical hysteresis depth (+1/−1 random walk)
  — depends on per-sample ENR estimator noise floor, NOT wall-clock.
- `hold_duration`: NE-state dwell — part wall-clock (phoneme) + part
  downstream NE-gain behaviour coupling.
- `noise_floor_hold`: room-noise adapt rate — wall-clock, but quiet-
  room cohort favours slower adapt (= less false-positive on speech
  transients).

Existing values (12 / 50 / 50) kept as empirically-validated cohort
tuning. Verdict: docs/v3_21_4_time_domain_audit_verdict.md.

#### Companion structural refactor: ms-based time-domain config

Renamed bare-int counter fields to ms-based fields so wall-clock
semantics auto-scale with hop_size at construction:
- `DominantNearendConfig.hold_duration_ms: int = 500` (was
  `hold_duration: int = 50` hops)
- `EchoModelConfig.noise_floor_hold_ms: int = 500` (was
  `noise_floor_hold: int = 50` hops)
- `trigger_threshold` kept as samples (dimensionless statistical
  hysteresis, NOT wall-clock-anchored).

`_DominantNearendDetector` + `ResidualEchoEstimator` now derive their
hop counts via `ms_to_hops()` at construction. `SuppressionGain` +
`ResidualEchoEstimator` accept `hop_size` alongside `sr` (the existing
v3.21.2 U3 sr threading). At 16k/10ms default, derived values are
50 / 50 — **byte-equal preserved**.

### U4.A — Per-bin H_error refresh retest CLOSED FAIL

Cherry-picked v3.21.1's per-bin H_error refresh substrate (commit
`d4e266e` → `12297ed`) onto canonical state and flipped flag to True.

Bucket means within ±0.007 dB (essentially flat), BUT cohort tail is
bad:
- 82 / 800 (10%) cases Δecho < −0.05
- 54 / 800 (7%) cases Δdeg < −0.05
- Worst single-case Δdeg **−0.437** (`LHsrJBRGnUKiMC2m` DT_static)

Same Pareto-damage pattern as v3.21.1 original verdict — bucket means
hide per-case damage. Root cause unchanged: AEC3 per-bin leakage
formula needs companion `ScaleFilter` + `FilterMisadjustment`
stabilisers we don't have aligned.

`use_per_bin_h_error_refresh: bool = False` (default OFF restored).
Substrate code retained as dormant research path for v3.22+.
Verdict: docs/v3_21_4_u4a_per_bin_h_error_retest_verdict.md.

### U4.B — B3 `lf_endpoint_hz` intermediate values CLOSED FAIL

Tested intermediate values between baseline (500 Hz) and B3-failed
(2000 Hz): 1000 Hz (U4.B1) and 1500 Hz (U4.B2). Both regress
monotonically:

| Variant | DT_st Δdeg | FS_st Δecho | Cohort tail |
|---|---:|---:|---:|
| 500 Hz baseline | 0 | 0 | 0 |
| **1000 Hz** | **−0.012** | **−0.007** | 21 echo + 19 deg |
| **1500 Hz** | **−0.015** | **−0.008** | 26 echo + 22 deg |
| 2000 Hz (B3) | −0.016 | −0.012 | (similar) |

Mechanism re-confirmed: 500-2000 Hz band on this cohort carries more
echo than voice on average. Wider LF sum → ENR rises → fewer NE
triggers → DT speech sees less protection.

`lf_endpoint_hz = 500.0` confirmed cohort-empirical sweet spot. B3
fully closed across all tested intermediate values.
Verdict: docs/v3_21_4_u4b_b3_intermediate_verdict.md.

### ReverbDecayEstimator — CLOSED NOT-PORTING

Audit across 4 representative cases shows our simpler 139-LOC port
NEVER adapts the decay value: stays at `default_decay=0.85` on all
2188-4307 frame cases. Four contributing factors:

1. **Architectural granularity mismatch** — `n_partitions=6` with
   `K_EARLY_REVERB_MIN_SIZE_BLOCKS=3` leaves only 0-2 data points for
   the slope regression on typical delays. AEC3 has 13 (64-sample)
   blocks for the same filter; partition-level port is fundamentally
   too coarse.
2. Upstream gates (`FilteringQualityAnalyzer` 4-gate AND +
   `StationarityEstimator`) intermittently block.
3. hygiene fix #2 recreate-on-recovery (v3.21.3) wipes estimator state on
   `_reset_filter_derived_state` events.
4. Even when gates open, regression has too few data points to produce
   stable slope.

Full AEC3 port (270 LOC: `AnalyzeFilter` + `EarlyReverbLengthEstimator`
+ `LateReverbLinearRegressor` + validation gates) wouldn't change
observable behavior without first addressing factors (2) + (3). The
constant `0.85` IS the operating reverb value (= non-adaptive fallback)
across all v3.21.x cycles.

v3.22+ prerequisites for re-attempting port: loosen upstream gates +
move estimator state to render-side preservation. Verdict:
docs/v3_21_4_reverb_decay_audit_verdict.md.

### 800-case AECMOS

No production change vs v3.21.3. Default-config byte-equal preserved.
Cumulative numbers vs v3.21.0 baseline unchanged from v3.21.3.

### Commits

- `d7698cd` — V4 time-domain audit CLOSED NOT-A-BUG.
- `12297ed` — v3.21.1 per-bin H_error substrate cherry-pick (default OFF).
- `0446287` — U4.A retest CLOSED FAIL + ms-based config refactor.
- `a80f5a0` — U4.B B3 intermediate values CLOSED FAIL.
- `136ee4c` — ReverbDecayEstimator audit CLOSED NOT-PORTING.

### Carry-over status after v3.21.4

All 4 v3.21.2 plan carry-overs now adjudicated:

| Item | Status |
|---|---|
| Time-domain unit-conversion bugs | CLOSED NOT-A-BUG (kept empirical values + ms refactor) |
| Per-bin H_error refresh retest | CLOSED FAIL (substrate dormant for v3.22+) |
| B3 intermediate values | CLOSED FAIL (500 Hz confirmed) |
| ReverbDecayEstimator full port | CLOSED NOT-PORTING (dormant in our pipeline; v3.22+ prerequisites needed) |

v3.22+ open work documented in each verdict; not blocking current
production.

---

## [3.21.3] — 2026-05-20 — source hygiene cycle (reset() AEC3 post-state + return_res_context + dead-knob removal)

**Headline**: 4 hygiene findings from source audit of v3.21.2, all
fixed and confirmed correct. Three are pure correctness improvements
(reset path completeness + documented contract implementation); one
is dead-code removal. One fix (hygiene fix #2) produces a measurable Pareto
shift on the 800-case bench — accepted as honest correction of a
previously-illusory FS_echo advantage.

### hygiene fix #1 (HIGH) — `AEC.reset()` clears AEC3 post-state

Pre-fix: `AEC.reset()` body initialised the v3.21 AEC3-aligned
post-stage fields (`_aec3_state` / `_aec3_ree` / `_aec3_sg` / OLA buf /
noise PSD / CN gain / pending events / stationarity tracker) in
`__init__` but never touched them on reset(). Re-using an AEC instance
across utterances carried previous-stream post-filter state into the
next stream.

Fix:
- Store `n_bins` + the env-var-driven `_sg_config` as instance
  attributes so reset can rebuild the post chain with the same config.
- Add `_reset_aec3_post()` helper. AecState + SuppressionGain don't
  expose in-place `reset()` so they're recreated; ResidualEchoEstimator
  and StationarityEstimator do, so they're called. Numpy buffers
  zero-filled, counters cleared.
- `reset()` body invokes `self._reset_aec3_post()` at the end.

Test coverage: 5 new unit tests in [python/tests/test_aec_reset.py](python/tests/test_aec_reset.py),
all PASS.

### hygiene fix #2 (MED) — `_reset_filter_derived_state()` clears AEC3 post chain

Pre-fix: helper cleared the legacy ResFilter post-state (gain_smooth /
echo_psd / noise_psd / gates) but v3.21.0 retired ResFilter. The
helper was never updated to clear the AEC3 post chain. Result: on
delay_first / delay_shift / p3h_diverged recovery, the filter taps
reset to zero but the AEC3 post chain kept its prior ERLE / R² / ERL
estimates — applying confident suppression logic on the noisy output
of a freshly-reset (un-trained) filter.

Fix:
- Update docblock to drop the stale ResFilter reference and add the
  AEC3 post chain to the CLEARED list (`_aec3_stationarity` +
  render-side counters in PRESERVED — render activity is input-side).
- Add `preserve_render_side` kwarg to `_reset_aec3_post()` so the same
  helper covers both AEC.reset() (full clear) and
  `_reset_filter_derived_state()` (preserve render-side).
- Invoke `self._reset_aec3_post(preserve_render_side=True)` in the
  helper body.

### hygiene fix #3 (MED) — implement `return_res_context=True` contract

Pre-fix: `AecConfig.return_res_context=True` was documented (CLAUDE.md
"Diagnostic surfaces") to switch `process()` return type from
`ndarray` to `(output, AecResContext)`, but `_res_context` was always
`None` so the documented contract never fired. Dead surface.

Fix: when `config.return_res_context=True` and the AEC3 chain ran,
populate `_res_context = AecResContext(...)` from in-scope state
(raw_output / echo_spec / far_spec / near_spec / far_power / converged
flag / erle_factor / dt_indicator / divergence / over_sub /
saturation / erl_estimate). The end-of-`process()` existing branch
`if _res_context is not None: return (result, _res_context)` now
fires, satisfying the documented contract.

Default path (`return_res_context=False`) byte-equal to v3.21.2 HEAD
on 25-case byte-equal sample.

Test coverage: 2 new unit tests, both PASS.

### hygiene fix #4 (MED) — remove dead legacy delay knobs

Two AecConfig fields became silent no-ops when v3.21 replaced the
legacy DelayEstimator with `LegacyDelayShim` wrapping the AEC3
estimator:

(1) `mov_rate_delay_est_enabled` + `delay_est_period_s_fast` +
    `delay_est_alpha_fast` — orchestrator wrote to
    `LegacyDelayShim._period_samples` / `_alpha` under EPC motion. The
    shim documents these as "no-op compat attributes"; its
    `accumulate()` doesn't read them. Pure dead writes.

(2) `trace_delay_est` + `trace_delay_est_path` — passed as `trace=`
    kwarg into `LegacyDelayShim`, which collected it into
    `_legacy_kwargs` and never consumed it. Documented to populate
    `aec.delay_est._trace_rows`, but the AEC3 estimator doesn't expose
    any such surface; the `--trace-delay-est` CLI flag was a silent
    no-op.

Removed: 5 config fields, the 17-line dead conditional in
orchestrator, 2 env var hooks in eval, 5 reference sites in
run_one_case. Byte-equal preserved (removed code was provably dead).

### 800-case AECMOS vs v3.21.2 baseline

| Bucket | Δecho | Δdeg | Note |
|---|---:|---:|---|
| FS_static | **−0.050** | +0.000 | Pareto cost of hygiene fix #2 (was illusion) |
| FS_movement | **−0.037** | +0.000 | Pareto cost of hygiene fix #2 (was illusion) |
| DT_static | −0.028 | **+0.025** | Pareto gain of hygiene fix #2 (speech recovered) |
| DT_movement | −0.032 | **+0.026** | Pareto gain of hygiene fix #2 (speech recovered) |
| NE | +0.000 | +0.000 | flat |

Mechanism (Pareto attribution): hygiene fix #1 / #3 / #4 are not exercised
on a fresh-instance-per-case bench so contribute no delta. hygiene fix #2
is the source. Before hygiene fix #2 (buggy): on filter recovery, AEC3 post
chain held stale ERLE / R² → applied confident-suppression logic to
noisy untrained-filter output → over-suppressed FS_echo (good metric)
but also over-suppressed DT speech (bad metric). After hygiene fix #2
(correct): post chain resets alongside filter → no fake confidence →
suppression backs off until filter retrains → less FS_echo gain, more
DT speech preserved. Pareto shift reveals the bug was extracting
illusory FS_echo at DT_deg cost.

### Cumulative 800-case AECMOS vs v3.21.0 baseline (a537b65)

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.197 | +0.000 |
| FS_movement | −0.154 | +0.000 |
| DT_static | −0.077 | **+0.119** |
| DT_movement | −0.080 | **+0.141** |
| NE | +0.000 | +0.003 |

### Commits

- `81a5103` — hygiene fix #1 AEC.reset() AEC3 post-state.
- `b2491a8` — hygiene fix #2 _reset_filter_derived_state() AEC3 post chain.
- `80da109` — hygiene fix #3 return_res_context contract.
- `fd2cfcd` — hygiene fix #4 dead legacy delay knobs removed.

---

## [3.21.2] — 2026-05-20 — Frequency-canonical bin-index alignment (HF damage Pareto step) + FS recovery

**Headline**: the v3.21 SuppressionGain port (b5728e5) copied AEC3
bin-index constants directly without converting for our 4× finer FFT
(AEC3 uses fft=128 / 125 Hz per bin; we run fft=512 / 31.25 Hz per
bin). Every HF-processing knob therefore landed at 1/4 of the intended
frequency — most damaging, the HF cap (`limiting_gain_band`) started
at **937 Hz instead of 3750 Hz**, slicing F2/F3 of voiced speech and
producing the user-reported Chinese /i/-vowel distortion ("低頻還在,
400 Hz 以上就被砍").

### Phase A — refactor (mechanism only)

Refactor 4 `SuppressionGain` config dataclasses from bin-index `int`
fields to frequency `float` fields, derive bins at use-site so values
auto-scale with `fft_size`:

- New [`python/modules/freq_utils.py`](python/modules/freq_utils.py):
  `hz_to_bin(hz, n_bins, sr=16000)` / `bin_to_hz(bin, n_bins, sr=16000)`.
  `fft_size` derived from `n_bins` so callers only thread the spectrum
  array, not FFT size separately.
- `HighFrequencySuppressionConfig`: `limiting_gain_band` →
  `limiting_gain_freq_hz`; `bands_in_limiting_gain` →
  `limiting_gain_width_hz`.
- `SuppressorConfig`: `last_lf_band` / `first_hf_band` /
  `last_lf_smoothing_band` → `*_freq_hz`.
- `EchoAudibilityConfig`: add `lf_band_end_hz` / `mf_band_end_hz`
  (audibility weighting band split; previously hardcoded bin 3 / 7).
- `DominantNearendConfig`: add `lf_endpoint_hz` (LF sum window for
  nearend detection; previously hardcoded `min(16, n)`).
- Consumers (`_limit_hf_gains` / `_weight_echo_for_audibility` /
  `_DominantNearendDetector.update` / `SuppressionGain.__init__`)
  resolve bins via `hz_to_bin()` against the input spectrum size.

Smoke-test confirmed all freq defaults reverse-compute to the
pre-refactor bin values (Phase A is mechanically byte-equal-at-init).
[P52 regime tests](python/tests/test_p52_regime.py) 18/18 PASS.

### Phase B — flip to AEC3 frequency-canonical (ship candidate)

After Pareto sweep across each unit-conversion knob, ship candidate
applies four of five flips; one was reverted as cohort-pareto-regressing:

| Knob | Old (bin / freq @ fft=512) | New (freq / bin) | Status |
|---|---|---|---|
| HF cap `lgb` | bin 30 / 937 Hz | **4000 Hz / bin 128** | SHIP |
| HF cap `biq` | 5 bins / 156 Hz | 156 Hz / 5 bins | SHIP (count-preserved; biq=625 Hz tested wash) |
| Mask `last_lf` | bin 5 / 156 Hz | **625 Hz / bin 20** | SHIP |
| Mask `first_hf` | bin 8 / 250 Hz | **1000 Hz / bin 32** | SHIP |
| Mask `last_lf_smoothing` | bin 5 / 156 Hz | **625 Hz / bin 20** | SHIP |
| NE detector `lf_endpoint` | bin 16 / 500 Hz | (kept 500 Hz) | **REVERT** — see below |
| `conservative_hf` inline | bins 20/29 / 625-906 Hz | 2500-3625 Hz | inline (flag-OFF, no-op) |

NE detector LF endpoint flip (500 → 2000 Hz, = AEC3 canonical bin 64)
was tested as T2 and regressed both DT and FS on the 800-case cohort
(DT_static deg −0.016 vs T1, FS_static echo −0.012). Cause: on this
cohort, the 500-2000 Hz band carries more echo than voice energy on
average, so widening the sum pushes `enr` higher and reduces nearend
triggers → cap fires more often → DT damage. AEC3 canonical alignment
does not always translate to cohort improvement.

### Phase C — FS recovery (U5.3)

Bumped `EpStrengthConfig.default_gain` from 0.014 → 0.020 in
[python/modules/residual/residual_echo_estimator.py](python/modules/residual/residual_echo_estimator.py).
`R²` in the nonlinear path = `X² × default_gain²`, so this scales R²
by `(0.020/0.014)² ≈ 2.04×` in nonlinear-mode frames.

S1 trace shows the 800-case cohort runs the nonlinear path 66-92% of
frames (linear ERLE not yet converged on FS), so this knob has high
population leverage. AEC3 precedent: `WebRTC-Aec3EchoPathGain` Aggressive
field-trial profile uses 0.02 — within AEC3-documented range, not an
invention.

Asymmetric Pareto-positive: every bucket non-negative vs Phase B T1.
See docs/v3_21_2_u5_fs_recovery_verdict.md
for full mechanism + U5.1 (mask_hf.enr_transparent) and U5.2
(normal_render_limit) closed-no-effect results.

### sr threading (U3)

`SuppressionGain.__init__` now accepts `sr=16000`; threads through
`_DominantNearendDetector` / `_weight_echo_for_audibility` /
`_limit_hf_gains` and all `hz_to_bin()` call sites. Orchestrator passes
`self.config.sample_rate` at construction. 16 kHz behaviour byte-equal;
verified sr=48000 now resolves lgb=4000 Hz to bin 43 (vs bin 128 @
16 kHz). See docs/v3_21_2_bin_audit_verdict.md
for the broader audit-clean verdict across `filter/`, `state/`, `delay/`,
`render/`, `epc`, `orchestrator`.

### 800-case AECMOS vs v3.21.0 (a537b65) baseline — final v3.21.2 (T1 + U5.3)

| Bucket | n | baseline echo / deg | new echo / deg | Δecho | Δdeg |
|---|---:|---|---|---:|---:|
| FS_static | 169 | 3.729 / 4.999 | **3.582** / 4.999 | −0.147 | +0.000 |
| FS_movement | 131 | 3.626 / 4.999 | **3.509** / 4.999 | −0.117 | +0.000 |
| DT_static | 186 | 4.237 / 2.387 | 4.188 / **2.481** | −0.049 | **+0.094** |
| DT_movement | 114 | 4.215 / 2.371 | 4.166 / **2.485** | −0.048 | **+0.115** |
| NE | 200 | 4.998 / 4.052 | 4.998 / 4.054 | +0.000 | +0.003 |

DT formant fidelity recovered (matches user HF damage report) at the
cost of HF echo cap relaxation in FS. FS regression remains net negative
but ~5% improved via U5.3 vs unmitigated T1.

### Audit verdicts

- docs/v3_21_2_audio_analysis_verdict.md —
  U1 quantitative band-energy analysis on 5 worst-deg DT_static cases.
  F2-F3 preservation **+0.48 dB mean** (all 5 cases positive +0.18–+0.91 dB);
  F1 +0.32 dB mean. PASS — the AECMOS deg gain corresponds to real voice
  formant preservation, not a spurious metric move.
- docs/v3_21_2_bin_audit_verdict.md —
  U2 exhaustive grep + line-read audit of all `python/modules/` for
  FFT-scale unit-conversion bugs. AUDIT-CLEAN: no other HIGH-severity
  bin-index bugs in production-active code.
- docs/v3_21_2_u5_fs_recovery_verdict.md —
  U5 sweep verdict + ship-candidate selection.

### Commits

- `7e9e612` — Phase A refactor + all 5 canonical flips (T2 state).
- `f1ea92c` — Revert B3 NE detector flip (T1 ship candidate).
- `5b7bf1c` — U1 audio analysis verdict (F2-F3 +0.48 dB).
- `8b7de5c` — U2 bin-index audit closure: codebase audit-clean.
- `6a071c1` — U3 sr threading through SuppressionGain consumers.
- `c7481a4` — U5.3 default_gain 0.014 → 0.020 + U5 verdict doc.

### Known carry-overs (v3.21.3+; same AEC3-alignment arc)

- **Time-domain unit-conversion bugs** (3 HIGH severity, parallel pattern
  to the bin-index bug fixed in this version): `trigger_threshold`,
  `hold_duration`, `noise_floor_hold` ported as bare ints from AEC3
  4 ms blocks into our 10 ms hops → 2.5× longer time-equivalent than
  AEC3 intended. Direction of fix opposes FS recovery so deferred.
- **ReverbDecayEstimator partial port** (1/3 of AEC3 size; missing
  `AnalyzeFilter` + `EarlyReverbLengthEstimator` + validation gates).
- **source hygiene findings** (4 items, all verified): `AEC.reset()`
  doesn't clear AEC3 post-state; `_reset_filter_derived_state()`
  docblock stale; `return_res_context=True` dead contract; legacy
  delay knobs (`mov_rate_delay_est_enabled`, `trace_delay_est`) no-op.
- **Conservative_hf inline path** — semantics changed from 625-906 Hz
  to AEC3 canonical 2500-3625 Hz; `conservative_hf_suppression=False`
  default means flag-OFF byte-equal.

---

## [3.21.0] — 2026-05-19 — Retire legacy ResFilter; AEC3 chain becomes the production post-filter

**Headline**: v3.21 ships the AEC3-aligned `_aec3_post` chain
(`AecState` + `ResidualEchoEstimator` + `SuppressionGain` + per-bin
comfort noise + sqrt-Hann OLA synthesis) as the single production
post-filter. The legacy 9-stage `ResFilter` chain (~2 200 LOC) is
deleted. The 5-preset menu collapses to a single `BALANCED` preset
(other 4 deleted in R1; the legacy BALANCED was retired in R2).
Cumulative cleanup: −5 565 Python LOC + −32 341 docs lines across
16 commits, byte-equal verified at every step (25-case representative
sample, 5 per bucket at echo percentiles 0/25/50/75/100).

### Architecture change

Production post-filter migration:

```
v3.10.5 — v3.20:                     v3.21:
  ResFilter 9-stage chain ─►           _aec3_post() chain ─►
    stage 1 residual_echo_psd            StationarityEstimator
    stage 2 softgate_emr                 AecState (read-only ADT over
    stage 3 epc_dt_cap                     12 sub-analyzers)
    stage 4 quiet_mask                   ResidualEchoEstimator
    stage 5 3bin_smooth                    (linear / render-based
    stage 6 hf_cap                          + ReverbModel tail)
    stage 7 pre_temporal                 SuppressionGain
    stage 8 temporal smoothing             (Wiener + over-estimation)
    stage 9 noise floor + CNG            Comfort noise generator
                                         sqrt-Hann synthesis OLA
```

The AEC3 chain was developed in stages from v3.18 (Phase C.C AecState
substrate) through v3.20 (Phase A.1 delay subsystem + Phase B PBFDKF
wiring + Phase C residual). v3.21 promotes it from substrate to
production by retiring ResFilter and the `use_aec3_residual` flag.

Reference comparison: docs/architecture_v3_10_5_vs_v3_21_vs_aec3.html.

### Bench scores (800-case AEC Challenge, BALANCED)

| Bucket       |    n |  echo (↑) |  deg (↑) | vs AEC3 ref deg | vs AEC2 ref deg |
|--------------|-----:|----------:|---------:|----------------:|----------------:|
| FS_static    |  169 |     3.729 |    4.999 | — | — |
| FS_movement  |  131 |     3.626 |    4.999 | — | — |
| DT_static    |  186 |     4.237 |    2.387 | **+0.537** | −0.003 |
| DT_movement  |  114 |     4.215 |    2.371 | **+0.521** | −0.019 |
| NE           |  200 |     4.998 |    4.052 | **+0.602** | −0.048 |

Anchor scores at docs/bench/v3_21_3aadd2d_baseline/.

### Cleanup rounds (in order)

| Round | Commit | Summary | Net LOC |
|---|---|---|---:|
| Phase A | a24d154 | Baseline + 25-case byte-equal harness + 800-case anchor | +8 684 (test infra) |
| R1 | c07d428 | Delete MILD / SOFT / AGGRESSIVE / MAXIMUM presets | −117 |
| R2 | 6267de0 | Drop legacy BALANCED → rename `BALANCED_AEC3` → `BALANCED` | −89 |
| R3 | 97509c3 | Remove `use_aec3_residual` flag + 2 runtime gates + env hook | −18 |
| R4 | 28ef604 | Collapse if-AEC3 / else-ResFilter to single `_aec3_post` call site | −20 |
| R5 | ceb9ead | Prune dead local-var prep + dead `self.res` state writes | −98 |
| R6 | b63dcbd | Drop dead `_residual_est` readers + Arc G + Arc T blocks | −194 |
| R7 | 0532c57 + 651ccdd | Delete ResFilter chain | −3 302 |
| R8 | 8b51007 | Retire `legacy_state.py` + delete `diagnose_gcc_phat.py` | −624 |
| R9 | c677725 | Delete `legacy_delay.py` | −282 |
| R10a | 1f0bb7f | Drop legacy ResFilter config knobs | −63 |
| R10b-1 | 2176a11 | arc_g / arc_t dead state init + reset paths | −54 |
| R10b-2 + R10c | 9d92334 | Drop dead substrate flags + readers + env hooks | −357 |
| R11 | 09ad7a9 | Archive sweep | (renames) |
| R12 | df793ce | Drop python module orphans | −347 |
| R13 | 75dddf7 | Aggressive `docs/` + `docs/archive/` prune | −32 341 |
| Phase D-1 / D-3 | f60b6a5 + (this commit) | Doc + version bump + v3.21 rewrites | — |

### Closed substrate retired

- v3.14 Arc P + Arc R + Arc S-orth.A.
- v3.15 Arc M v1+v2+v3 / Arc G / Arc T.
- v3.18 Phase C.C AecState facade / C.D-α leakage_diverged / C.E + C.E
  branch ablations / D-γ retried mask shape swap.
- P52 Phase B subclass-and-delegate ResFilter refactor.
- P53 / P55 / P58 dual-filter / dual-PBFDKF / AEC3-pattern RES
  restructure (closed CANNOT-SHIP on 800-case during their respective
  cycles; substrate retained as research log until R13 doc cleanup).

Shipped substrate retained:

- v3.18 Phase A.2 shadow NLMS coarse filter (default ON).
- v3.18 Phase B.2 / B.3 FilterMisadjustmentEstimator + ScaleFilter
  (default ON v3.21).
- v3.18 Phase C.A FilterAnalyzer (audit-only).
- v3.18 Phase F.1 / F.3 AEC3 event classification + asymmetric reset.

### Tests + tooling

- `python/check_byte_equal.py` — 25-case representative byte-equal
  harness. Reference at `docs/bench/v3_21_3aadd2d_baseline/byte_equal_
  reference.json`. Must report `=== 25/25 PASS, 0 FAIL ===` before any
  commit that touches Python outside docs.
- `python/test_f3_1_mic_excess.py` retired with ResFilter (R7).
- `python/tests/test_p52_regime.py` retained — enforces the
  `AcousticRegimeClassifier` anti-loophole contract.
- `python/diagnose_gcc_phat.py` retired (R8) — research-only.

### Docs

Canonical doc set at `docs/` root collapsed from 66 → 11:

- `aec_methods.md` (v3.21 rewrite — algorithm spec).
- `aec_algorithm_guide.html` (v3.21 rewrite — presentation overview).
- `architecture_v3_10_5_vs_v3_21_vs_aec3.html` (NEW — comparison).
- `pbfdkf_shadow_intro.md` / `dtd_design.md` (canonical algorithm refs).
- `c_user_and_integration_guide.md` (C API + integration).
- `refactor_modules_layout.md` (current module map; v3.21 rewrite).
`docs/archive/` retired entirely — 130+ per-arc verdict / design docs
deleted across R11 / R13 / the docs-trim that landed alongside Phase
D-2. The historical record lives in this CHANGELOG and in git history.

---

## [3.15.0] — 2026-05-15 — v3.15 arc closeout (Arc T detector default ON)

**Headline**: Zero ship-able algorithm changes; one preset default flip
(Arc T cohort tail real-time detector → BALANCED default ON, byte-equal
on audio output). Six candidate arcs CLOSED CANNOT SHIP after exhausting
their structural ceilings. Six default-OFF substrate flags retained for
v3.16 retry. v3.16 RES refactor plan authored with 13 ranked candidates
(5 with predicted Δ ≥ +0.005); v3.16 cycle authorised pending phase
kickoff.

### Production-affecting (BALANCED preset behaviour)

- **§10.S0b** (`5bb2fa8`): `arc_t_cohort_detector=True` in BALANCED.
  Cohort tail real-time detector populates `AecStats.cohort_tail_T`
  per-frame and writes `self._arc_t_cohort_tail_signal` field. All
  consumers (5 `arc_m_t_gated` gates + 1 RES preempt path) require
  additional default-OFF flags, so detector ON is **byte-equal on audio
  output** — only diagnostic state changes. 5/5 sanity case byte-equal
  PASS (NE / DT / DT_movement / FS / FS_movement, atol=0.0).
  - Why: enables v3.16 RES refactor consumers (Phase 3 candidates
    v3.16-A force_render OR-in / v3.16-B ENR-path lift) to read the
    signal without per-bench env-flag flipping.
  - Verdict: docs/v3_15_arc_t_s1_design_and_verdict.md

### Bug fixes shipped

- **§1.0.S1 B4** (`3860335`): drop dead `'converged'` branch in
  quiescent re-sync (`_prev_filter_state` checks). The string belonged
  to `AecFilterState` enum vocabulary, not the internal P3f state
  machine — the branch was structurally unreachable. Cleanup removes a
  code-clarity hazard; behaviour byte-equal on production paths.
  - Verdict: docs/v3_15_b4_verdict.md
- **§1.0.S2 B5** (`bb9076f`): `_shadow_copy_err_baseline` doc aligned
  with actual implementation as RESERVED (declared but not wired —
  future arc scope). Doc-only change.
  - Verdict: docs/v3_15_b5_verdict.md
- **§10.S0c B9** (`1323f92`): bench tooling `--workers` CLI flag +
  per-scenario chunk-split (`n_chunks = workers // 3`); 800-case bench
  ~2× speedup over hardcoded `max_workers=3`. Byte-equal sanity 120/120
  between j=3 and j=6 outputs.
- **§1.5b naming** (`03e311b`): renamed `arc_m_v3_t_gated_enabled` →
  `arc_m_t_gated_enabled` per project naming convention (drop numeric
  version suffix from live config field names; keep arc-codename
  prefix as identifier).

### Closed CANNOT SHIP (no production change; default-OFF substrate retained)

- **§1.2 DT-NE compression fix** (`81f59bf`): per-state ENR + per-bin
  override candidates (full + per-bin only). Both fail FS Δecho bars
  3.8–10× over. Same family as v3.13 E5: filter-protection mechanism is
  trade-off-bound. Substrate `dt_ne_compression_fix=False` retained.
  - Verdict: docs/v3_15_dt_ne_compression_fix_closure.md
- **§1.4 Arc M V1+V2** (`92f264b`): EPC-gated per-band Kalman Q boost.
  V1 (0.5/1.0/2.0) FS_movement −0.027; V2 (0.7/1.0/1.5) cohort tail
  −0.053. EPC ⊃ cohort tail catastrophe windows — boosting Q during
  EPC-active windows boosts Q during catastrophe windows. Substrate
  `arc_m_epc_gated` retained.
  - Verdict: docs/v3_15_arc_m_closure.md
- **§1.4 Arc G** (`acd2f2d`): per-band W reset on detected gain-change
  drift. ERLE Δ=−1.48 dB / 0/5 audible improvement on listen cohort.
  Destructive zero-out; v3.16 candidate C8 considers non-destructive
  partial decay. Substrate `arc_g_per_band_w_reset` retained.
  - Verdict: docs/v3_15_arc_g_closure.md
- **§1.5 Arc T S2 RES preempt wiring** (`3d77486`): two independent
  no-op bugs proven by single-case smoke test on `qNvSMyU` (output
  bit-equal ON vs OFF):
    - **H1** (`over_sub × 1.3`): DEAD CODE in BALANCED — `over_sub`
      only read by `gain_type='wiener'`; all 5 presets use `'enr'`.
    - **H2** (`_using_render_based = True`): OVERWRITTEN 1 line later
      by `_residual_est.compute_residual_echo()` state machine.
  Substrate `arc_t_res_preempt_mode` retained for code symmetry; v3.16
  candidates v3.16-A / v3.16-B fix the integration patterns.
  - Verdict: docs/v3_15_arc_t_s2_wiring_closure.md
- **§1.5b Arc M.v3 T-gated rescue** (`03e311b`): wraps 5 `_arc_m_q_boost`
  call sites with `(arc_m_t_gated_enabled AND _arc_t_cohort_tail_signal)`
  gate. Subset 60-case bench: V1 ΔERLE_lin == M.v3 ΔERLE_lin in EVERY
  bucket EVERY decimal (linear filter byte-equal C1 vs C2). Per-case
  MD5 verified on `qNvSMyU` — ours.wav identical between V1 and M.v3.
  Trace: 4/5 q_boost fires at signal=False (rising-edge events fire AT
  boundary of signal assertion); 1/5 at signal=True was on shadow
  filter (S-orth.A decoupled, no main-output path). Structural
  timing/scope mismatch — discrete-event signals don't pair with
  persistent-state signals without designed temporal alignment.
  Substrate `arc_m_t_gated_enabled` retained. v3.16 candidate C7
  documents 3 retry options (α predictive signal / β post-assertion
  hysteresis / γ per-filter dispatch).
  - Verdict: docs/v3_15_arc_m_v3_closure.md
- **§1.6 Arc F per-band Kalman Q schedule** (`415e8ec`): cohort tail
  damage. Substrate `kalman_q_per_band` + `kalman_q_band_scales`
  retained — paired with Arc M V1 substrate (V1 reproduction needs
  THREE flags atomically).
  - Verdict: docs/v3_15_arc_f_closure.md

### Audited but produced no actionable work

- **§1.7 RES audit** (`04c1dfe`): 60-case directional audit on the
  v3.15 closeout substrate. Headline finding: `ne_g_floor` fire-rate
  0.93 → **0.000 on DT** — v3.14 Arc P + R raise `spectral_g_min`
  enough that the `max(spectral_g_min, ne_g_floor)` comparison never
  picks `ne_g_floor`. v3.13 verdict's "universal baseline floor" no
  longer holds on v3.15 substrate. Adds NEW v3.16 candidate **C1b**
  (`ne_g_floor` removal) alongside C1 (`epc_dt_cap` removal — still
  0/all-buckets, doubly dead).
  - Sample bias: `--n-cases 60` enumerated alphabetical-first cases,
    all in `doubletalk/` scenario (40 DT_static + 20 DT_movement);
    0 FS / 0 NE / 0 cohort_tail. 800-case re-audit at v3.16 phase
    entry mandatory.
  - Audit + plan: docs/v3_15_res_audit_and_refactor_plan.md

### v3.16 candidate plan (13 candidates, 5 phases, 21–30 sprints)

| Phase | Candidates | Sprints |
|---|---|---|
| 0 housekeeping | HK-1 (B3 CNG seed), HK-2 (pcb1N patch), C1 (epc_dt_cap removal), **C1b (ne_g_floor removal — substrate-shift)** | 3 – 4 |
| 1 foundation | C5 (per-state RES interface), C6 (DelayEst audit ⭐ critical gate) | 4 – 5 |
| 2 RES refactor | C2 (ENR per-state × per-band), C3 (4-cap reorder), C4 (noise_floor / CNG) | 6 – 9 |
| 3 Arc T consumers | v3.16-A (force_render OR-in), v3.16-B (ENR-path lift) | 4 – 6 |
| 4 Arc M / G retry | C7 (Arc M.v3 α/β/γ), C8 (Arc G non-destructive decay) | 4 – 6 |

**C6 DelayEst audit** is a critical gate — 5 movement-related v3.15
closures (cohort tail, Arc M V1 FS_movement, Arc F cohort tail, Arc G
destructive W reset, §1.1 H5 DT-NE hypothesis) share echo-path-changing
substrate where DelayEst tracks. If audit confirms DelayEst is the
upstream cause for ≥ 30 % of those wall magnitudes, Phase 3-4 ROI
estimates change.

### Inherited debt (carried to v3.16)

- **v3.13 E2 Path 3 DT debt** (DT_static −0.050, DT_movement −0.025):
  remains unrecoverable in v3.15 production. Closure target moves to
  v3.16 RES refactor (C2 / C4 / C3 totalling +0.005 to +0.040
  predicted DT bucket recovery).

### References

- Top-level closeout: docs/v3_15_closeout_verdict_pack.md
- v3.16 plan: docs/v3_15_res_audit_and_refactor_plan.md
- All v3.15 closure / verdict docs: [docs/v3_15_*.md](docs/)

---

## [3.14.0] — 2026-05-14 — v3.14 arc (per-band ERL/ENR + decoupled shadow)

**Headline**: Three production changes ship to BALANCED — Arc P
(adaptive per-band ERL EMA), Arc R (per-band ENR thresholds with
`block_lf` tilt), Arc S-orth.A (decoupled shadow Kalman state). First
mechanism in 5+ shadow-retirement attempts that produces genuinely
independent shadow Kalman state. Arc H (Huber loss) closed CANNOT
SHIP after H.S1 — real listen mic saturation is bounded NL residual
floor, not impulsive gradient spike. Arc D (filter-state-aware RES
policy) substrate shipped on `feature/v3.14-arc-d` but not merged
(deferred to v3.15 then v3.16).

### Production-affecting (BALANCED preset behaviour)

- **Arc P P.S3** (`9162d78`): adaptive per-band ERL EMA driven by
  `error_psd / far_lw` (Option B source signal). Replaces scalar
  `erl_estimate=0.3` (7× over-estimate in low-coupling rooms) with
  3-band LF/MF/HF EMA (α=0.99). Flag `f3_1_per_band_erl_adaptive=True`.
  - Verdict: docs/v3_14_p_s3_verdict.md
- **Arc R R.S2** (`5e3e96b`): per-band ENR thresholds with `block_lf`
  tilt (raise LF, lower HF). DT bucket +0.007 dB mean Δdeg on
  800-case; FS regression within −0.02 bar. 7-case xrtntuju listen
  verification: NE not damaged, FS not audibly leaking. Paired with
  `f3_1_per_band_erl_adaptive` for end-to-end per-band gate. Flag
  `res_per_band_enr=True`. R.S2.1 admit_hf control later confirmed
  block_lf winner direction; FS_static intrinsic cost is per-band ENR
  mechanism overhead, not direction-dependent.
  - Verdict: docs/v3_14_r_s2_verdict.md
- **Arc S-orth.A** (`8089974` + `f08ddbf`): decouple shadow's Kalman
  `_error_psd` + `R` from main's. 800-case GREEN PASS — all 5 buckets
  within bar; cohort tail `qNvSMyU` Δecho +0.0036; state correlation
  drops main vs shadow 0.99 → 0.47 on DT_static (target 0.5–0.7 hit).
  Includes Option B quiescent re-sync safety regularization (10% blend
  toward main when 3× drift in steady FS). Flag
  `shadow_state_decoupled=True`.
  - Verdict: docs/v3_14_s_orth_a_s2_verdict.md
- **Housekeeping B1 + B2** (`5fbceb0`): `PBFDKF.reset()` cleanup
  (unconditional `delattr` of `_p_max_override_frames`); `AecStats`
  `filter_state` enum/string contract aligned at API boundary.

### Closed CANNOT SHIP (substrate retained)

- **Arc H Huber loss** (`feature/v3.14-arc-h` HEAD): synthetic
  clipping (19.8% bursts) Huber δ ≥ 0.30 identical to L2 (no clipping
  trigger), smaller δ degrades. Real listen cases (01/02/07): Huber
  strictly worse than L2 for every δ. Impulse spike test confirms
  Huber works for true impulsive outliers — but real listen mic
  saturation = bounded NL residual floor (model mismatch), NOT
  impulsive gradient spike. Same physics wall as v3.13 E4/E5
  amplitude-domain closures. Substrate
  `tools/research/v3_14_h_s1_huber_proto.py`
  preserved.
  - Verdict: docs/v3_14_h_s1_verdict.md

### Substrate shipped but not merged to BALANCED

- **Arc D filter-state-aware RES policy** (`feature/v3.14-arc-d`
  HEAD `0218906`): per-state ENR tuples + 4-cap on/off. 800-case
  bench Δ ≈ 0 on aggregate (only `suspicious_dt + diverged` states
  differentiate — rarely fire in production). Deferred to v3.15
  (which deferred it to v3.16 C2 candidate that subsumes Arc D's
  `coarse_learning` tuple into per-state × per-band ENR refactor).

- **Arc S-orth.B** L1-regularized shadow weight update
  (`feature/v3.14-arc-s-orth-b`): bucket means within hard abort bars
  (FS Δecho −0.013, DT Δdeg +0.000~+0.003) BUT two new large per-case
  FS outliers (`0KjzXA3g…` FS_static Δecho −1.557; `KSN5Jrzo…`
  FS_movement Δecho −0.704). NOT promoted; substrate retained for
  potential v3.15 / v3.16 S-orth.B.S3 retry.
  - Verdict: docs/v3_14_s_orth_b_s2_verdict.md

### Volterra arc (research substrate, not what shipped as v3.14)

`feature/v3.14-volterra` carried the Volterra non-linear inverse arc
(S1 cohort baseline + S2 detector wiring + S3.0 joint Hammerstein
feasibility PASS, +2.99 dB mean ERLE on 5/5 NL). Branch was deleted
in v3.15 closeout cleanup; design lock + S2 audit + S3.0 verdict docs
preserved under [docs/v3_14_volterra_*.md](docs/). Volterra arc remains
listed as v3.16 Track 2 in the v3.15 plan §9 roadmap (re-authorisation
required if reopened).

### References

- Per-version evolution: docs/aec_v3_evolution.md §v3.14
- v3.14 plan archive: docs/v3_14_plan.md (if preserved)

---

## [3.13.0] — 2026-05-14 — v3.13 arc closure

**Headline**: Single production change shipped (E2 Path 3); two architectural
arcs (E4 NLP + E5 Saturation deepening) closed CANNOT SHIP after exhausting
their physics ceiling; back-end RES audit closed with limited refactor
surface. v3.14 Volterra design lock opens as the canonical breakthrough path.

### Production-affecting (BALANCED preset behaviour)

- **E2.S5 Path 3** (`5b1760c`): `eval_aec_challenge.py` `estimate_delay()`
  default `max_delay_ms` raised 250 → 1024 ms. Aligns bench pre-alignment
  with online F-DelayTrack search window. Closes 6/8 worst-FS listen cases
  that had residual delay 1200–10000 samples (75–625 ms) AFTER prior
  GCC-PHAT pre-alignment.
  - 800-case Δ vs v3.11.x baseline:
    - FS_static Δecho **+0.107**
    - FS_movement Δecho +0.018
    - DT_static Δdeg **−0.050** (accepted "RES unmasking" trade-off)
    - DT_movement Δdeg **−0.025** (accepted)
    - NE Δdeg −0.002 (within bar)
  - Listen: xrtntuju 5-clip DT regression 0 reg / 2 imp; cohort tail
    (qNvSMyU FS_static) Δecho −0.004 (within bar).
  - Trade-off deferred to v3.14+ per-state ENR refactor.
  - Verdict: docs/v3_13_e2_s5_verdict.md

### Closed CANNOT SHIP (no production change; default-OFF substrate retained)

- **E4 NLP arc** (`3e10621`): 12 sprints S1–S6b. SubtractiveNLP detector
  validated (5/5 NL cohort listen, 0% NE FP after S4.1 cancellation-ratio
  gate). Suppressor (harmonic-pinned σ=50 Hz Gaussian mask) PROVABLY
  ATTENUATES (voice formants disappear at g_min=−30 dB) but **NO AUDIBLE
  NL REDUCTION** at any aggression level (S6a/S6b listen). Closure
  mechanism: multiplicative spectral mask `m[k,t] · Y[k,t]` only modulates
  amplitude; real NL is dominantly phase distortion + time-domain
  transients — unreachable by any amplitude mask family.
  - Detector preserved as default-OFF (`e4_nlp_enabled`); reused in v3.14
    as NL-frame identifier component of ensemble.
  - Verdict: docs/v3_13_e4_s6_verdict.md +
    docs/v3_13_e4_s6a_s6b_verdict.md

- **E5 Saturation deepening arc** (`c871a5d`): 4 sub-variants (S2/S3/S4a/S4b).
  All on FS-vs-DT trade-off line, slope ~0.5 dB DT loss per +1 dB FS gain.
  All FAIL DT Δdeg ≥ −0.005 hard bar by 4–10×. Mechanism: amplitude-layer
  detector cannot distinguish FS-NL frames from DT high-echo frames — same
  correlation signature in [0.7, 0.95] mic-peak band fires on both.
  - Detector (E5.S3 mic-lpb correlation gate) preserved; reused in v3.14.
  - Verdict: docs/v3_13_e5_closure_verdict.md

### Audited but produced no actionable work

- **Phase 3 RES gain_floor 5-path audit** (`6cdfbb0`): Empirical fire-rate
  audit on 800-case BALANCED. Findings:
    - `epc_dt_cap`: 0/800 fires (DEAD CODE confirmed, removable)
    - `spectral_floor`: 97% on cohort tail qNvSMyU (LOAD-BEARING)
    - `ne_g_floor`: 88–99% all buckets, low skew 0.13 (Q7 V3 fragmentation
      hypothesis FALSIFIED — universal baseline floor, NOT main FS leak
      carrier)
    - `quiet_mask` / `divergence_floor`: physical fallback, KEEP
  - Canonical refactor surface SMALL (1 path removable, 1 absorbable);
    expected AECMOS delta ~ 0 (consistent with v3.12 5-NEUTRAL closure).
  - S6–S7 (canonical refactor) deprioritized; S8–S9 (4-cap audit + per-state
    ENR) deferred to v3.14+.
  - Verdict: docs/v3_13_phase3_res_audit_verdict.md

### v3.14 candidate items (deferred)

- **Volterra non-linear inverse filter (HIGHEST priority)**: 6+ month
  dedicated arc. Detector reuse from E4.S2 + E5.S3.
- Phase 3 RES canonical refactor (LOW, cosmetic)
- F-HFR per-band Q/R (LOW-MED, structural)
- E1 mic_dynamic_margin (LOW, 1 listen case)
- DT regression mechanism per-state ENR (MED)

### References

- Top-level closure: docs/v3_13_arc_closure.md
- v3.14 design lock: docs/v3_14_volterra_design_lock.md

---

## [3.12.x] — 2026-05-13 — Stage 1 RES exhaustion (NEUTRAL closure)

**Headline**: 5 NEUTRAL sprints (S6 / S6b / S7 / S10 / S11) targeting every
meaningful gate on ENR denominator and numerator. Stage 1 RES surface is at
local optimum — Δ ≈ ±0.001 on every bucket. No production change. Worst-FS
8-case listen redirected work to filter-side arcs (E1/E2/E4/E5), opening
the v3.13 plan.

### Notable

- Q3 / Q6 / Q7 RES architectural hypotheses fully falsified by 5-NEUTRAL +
  listen.
- v3.11.x retained as production ceiling.
- Verdict: docs/v3_12_s6_s11_stage1_locked.md

### Sprints

- S6 / S6b: nearend_floor refinement variants — NEUTRAL.
- S7: dt_per_bin unified (third Q7 V3 carrier) — NEUTRAL (docs/v3_12_s7_verdict.md).
- S8: noise_floor_psd dominant carrier diagnostic.
- S9: noise_floor_refine triple-trial null.
- S10: res_noise_floor_refined NEUTRAL ([docs/v3_12_s10_*.md](docs/)).
- S11: Cap2 FS-loosen NEUTRAL.

---

## [3.11.2] — v3.11 Phase 1 promotions, third tranche

### Production-affecting (BALANCED preset)

- `f_e1_enabled = True`: F-E1 ERL clip range extension + far_active hysteresis.
  - 800-case: NEUTRAL bench mean (Δ < 0.001), addresses extreme-ERL listen
    edge cases.
- `f_delaytrack_enabled = True`: F-DelayTrack continuous delay variance
  (replaces hard cut at confidence ≥ 0.5).
  - 800-case: NEUTRAL bench mean.

### References

- Phase 1 final review: docs/v3_11_phase1_final_review.md

---

## [3.11.1] — v3.11 Phase 1 promotions, second tranche

### Production-affecting (BALANCED preset)

- `shadow_mu_state_aware = True` (B6): 4-band shadow µ schedule with
  `suspicious_dt → 0.5` band; binary cut → state-aware.
  - 800-case bucket-mean +0.007 ΔERLE; wlAXM0i listen verified
    indistinguishable from baseline.

### References

- B6 listen verdict: [docs/v3_11_phase1_b6_listen_verdict.md](docs/) (see
  Phase 1 final review)

---

## [3.11.0] — v3.11 Phase 1 promotions, first tranche

### Production-affecting (BALANCED preset)

- `shadow_r_reset_enabled = True` (B5, Yang 2017 R-reset): symmetric R-reset
  on EPC (extends F2.3 to shadow filter's `_error_psd` + `R`).
- `f_e5_enabled = True` (F-E5 saturation 4-fix bundle):
  - mic soft-clip when sat_mic > 0.3
  - main mu sat-gate (freezes at sat_level > 0.5)
  - error_psd fast-attack reset on sat → clean transition
  - shadow_rise mask during saturation
  - sKXucFp4 single-case top: +0.348 dB Δecho
- `diverged_reset_enabled = True` + `diverged_reset_triple_and = True`:
  triple-AND gate (streak + shadow_advantage > 2.0 + filter_state == diverged)
  to avoid F2.2 EMA trap (which closed FAIL with 17 reg / 8 imp).

### Bench

- 5 buckets verdict OK; Δ < 0.001 dB vs v3.10.6 baseline; cohort tail
  qNvSMyU +0.010 linear preserved.

### References

- docs/v3_11_phase1_final_review.md
- F2.3 R-reset verdict: docs/f2_3_verdict.md
- F2.4 mu holdoff verdict: docs/f2_4_verdict.md

---

## [3.10.6] — three v3.10.6 fix promotes (2026-05-12)

### Production-affecting (BALANCED preset)

- **F3.1 v3** (mic-excess gate + dt_per_bin blend): per-bin NE evidence,
  AUROC 0.871. Closes xrtntuju 5-clip DT NE-damage regression cohort.
- **F2.3** (epc_r_reset_enabled): EPC R-reset for main filter (Yang 2017
  pattern, single-filter scope).
- **F2.4** (mu_holdoff_no_reset): release-counter form of `_simple_mu_holdoff`;
  prevents marginal-DT counter resets.

### References

- Plan closure: project_plan_hazy_lynx_closure.md
  (memory)
- F3.1 / F2.1 verdicts: project_f3_1_f2_1_results.md
  (memory)

---

## [3.10.5] — baseline reference (pre-v3.11 era)

The 800-case AECMOS reference snapshot used as the comparison baseline for
all v3.11+ work. Captured in `results/v3_10_5_main/scores.json`.

### Bucket means (800-case BALANCED)

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.646 | 4.999 |
| FS_movement | 131 | 3.705 | 4.999 |
| DT_static | 186 | 4.221 | 2.323 |
| DT_movement | 114 | 4.053 | 2.368 |
| NE | 200 | 4.998 | 4.011 |

---

## Aggregate v3.10.5 → v3.13.0 (this release vs pre-v3.11 baseline)

Computed from `results/v3_10_5_main/scores.json` vs
`results/v3_14_baseline/scores.json` (rendered today on v3.13 closure HEAD;
v3.14 detector substrate is default-OFF so render = pure v3.13 behaviour).

| Bucket | Δecho | Δdeg | Source |
|---|---:|---:|---|
| FS_static | **+0.107** | 0 | E2 Path 3 |
| FS_movement | +0.018 | 0 | E2 Path 3 + Phase 1 micro-effects |
| DT_static | +0.014 | **−0.050** | E2 Path 3 (RES unmasking, accepted) |
| DT_movement | +0.005 | **−0.025** | E2 Path 3 (accepted) |
| NE | 0 | −0.002 | NE invariant preserved |

**Net**: FS bucket improved (Δecho +0.107 / +0.018), DT bucket trade-off
(echo micro-up, deg micro-down within bar), NE unchanged. Cohort tail
listen materially improved (E2 Path 3 closes 6/8 worst-FS listen edge
cases; xrtntuju 5-clip 0 reg / 2 imp).

---

## Earlier history

For v3.10.4 and earlier (v3.7 → v3.10.4), see canonical research log
docs/SUMMARY.md. v3.7.1 is the most recent git tag
prior to v3.13.0; tags between v3.7.1 and v3.13.0 are P52/P53 milestone
tags rather than product versions:

- `p52-phase-a-closed-path3` (2026-05-12)
- `p52-phase-b-closed`
- `p53-design-locked`
- `p53-step-0-closed-T0E`
