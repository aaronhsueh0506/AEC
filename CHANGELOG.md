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
`docs/timing_constant_inventory.md` and the A/B sets that were under `eval/ab_evidence/`
(archived outside the repository since 2026-10-28; in git history through `6cf6785`).
List the retired ones with (this reports the two example paths above as well,
since they are themselves retired documents):

```bash
grep -oE '\bdocs/[A-Za-z0-9_/-]+\.(md|html)' CHANGELOG.md \
  | while read -r p; do [ -f "$p" ] || echo "retired: $p"; done
```

Versioning: `__version__` in [python/aec.py](python/aec.py) versions this
library's PUBLIC API and OUTPUT CONTRACT — the C header
`c_impl/include/aec.h` (struct layouts, entry-point signatures, context-field
semantics), the auto-derived default signal grid, and whether `aec_process()` output stays bit-identical. MAJOR bumps
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


## [Unreleased] — 2026-11-16 — `libaec.a` `__TEXT` sizes in the docs are re-measured on the default build

`c_impl/README.md` and `c_impl/STATIC_MEMORY.md` quoted the sizes measured before `AEC_LAYOUT_VERSION` 5 and before `AEC_STAGE_TIMING` became the default. Same method (`size` summed over the members, Apple clang, arm64, `make lib` with the same flags): `SIMD=1` 160,122 -> 167,730 B, `SIMD=0` 140,678 -> 147,482 B, `AEC_NO_ARRAY=1` 139,234 -> 143,686 B and 122,914 -> 125,694 B. With `PROFILE=0` the same builds are 167,086, 146,822 (`SIMD=0`), 143,134 B (`AEC_NO_ARRAY=1`, `SIMD=1`).

## [Unreleased] — 2026-11-16 — pyc-parity: `hard_reset.48k1024` lean and full register a T-01 tie

Test only. With the 48 kHz streak minimum at 8 hops the std-math C build and Python take one per-bin leakage decision differently in these two cells (hop 1387, bin 222: `e2_refined / e2_coarse - 1` is -1.6e-6 in C and +1.8e-6 in Python, the coarse operands differ by 3.7e-6 relative); Layer 1 reads 8.3e-4 against 3e-4. The shadow-copy counter, `main_paused`, baseline and pause countdown, the FORM selection (up to the hop that follows the divergence), the guard and the delay events and reverts (858, 1038) agree on every hop before it. Forcing that decision in Python drops the output difference to 1.6e-7; two more ties follow on the forced trajectory (hop 1388 bin 302, margins -2.9e-5 / +2.9e-5; hop 2146 bin 287, -2.2e-6 / +1.0e-5), after which no leakage or noise-gate decision differs on any hop. The shipping build is 1.7e-7 from Python on the same scene. Both cells are registered with `--force-leak 1387:222:1,1388:302:1,2146:287:1` (T-01 contract of 2026-10-28: hard seam gates green as run, every gate green forced, the cell fails if it turns green unforced or the forced run goes red); seven cells now carry T-01.

## [Unreleased] — 2026-11-16 — The shadow-copy rescue streak is 80 ms on every grid

The shadow-copy streak minimum (`sc_hys_streak_min()` in `epc_shadow.c`, `PathChangeRegimeHandler._streak_min_hops()`) is `ceil(sample_rate * 80 / (1000 * hop_size))` hops, computed in integers by both ports: 10 at 16 kHz hop 128, 5 at 16 kHz hop 256 and 8 kHz hop 128, 8 at 48 kHz hop 512 (it was 10 on every grid). Output changes on the other grids (16 kHz hop 256, 48 kHz hop 512); 16 kHz hop 128 is bit-identical; `docs/timing_constant_inventory.md` moves the entry to "Already retimed".

## [Unreleased] — 2026-11-16 — The fixed eval manifest keeps official clips only (61 cases)

`eval/manifest_90case_stems.txt` and `eval/manifest_90case.json` lose the 29 near-end stems whose clips exist only under `nearend_singletalk/_quarantine_non_official/`: evaluation data comes from the official sets and own recordings. 61 cases remain (30 far-end singletalk, 30 doubletalk, 1 near-end singletalk), 122 clips. The file names keep the `90case` label so the runners, the pin keys and the references in older entries still resolve.

- `eval/manifest_pin.py` resolves a stem under its scenario directory only; the quarantine fallback is gone, so a clip that sits anywhere else is reported as unresolved. `eval/manifest_90case.sha256` is rewritten from the corpus: the 122 remaining clip digests are the ones the old pin held, and the stems and JSON digests are new.
- `python/eval_manifest90.py` prints its case count instead of "90-case"; `run_c_ab.sh` and `run_py_ab.sh` count the stems from the file as before.
- The near-end bucket now has one case, so its SDR / `deg_mos` means are a single clip. A near-end bucket of useful size needs a new selection from the official near-end clips, which changes the frozen set and is not done here.

## [Unreleased] — 2026-11-16 — `AEC_STAGE_TIMING` is on by default

`aec_get_last_timing()` and `aec_array_get_timing()` report real per-hop costs in the default build. `aec.h` defines `AEC_STAGE_TIMING` as 1 when the build does not say otherwise, and the Makefile's `PROFILE` defaults to 1 and always passes `-DAEC_STAGE_TIMING=$(PROFILE)`; `make PROFILE=0` is the way to turn it off (every field reads 0, the archive references no `clock_gettime`). `PROFILE` stays in `CFG_SIG`. No ABI change: `sizeof(Aec)`, every offset and every sample are unchanged. The four baseline streams (`split_entry`, `split_entry2`, `aec_array`, `aec_array2`, `SIMD=1`) compare byte-identical between the build before this change and the new default build, `META` lines included.

- Cost: seven clock reads per mono hop, 18.5 ns each on the development host, about 0.13 µs per hop (0.3-0.9% of a 15-45 µs hop); the four-microphone array reads about thirty times per hop (counted from the stamps, not measured: about 0.6 µs). Interleaved A/B of `bench_rtf --scene bursts` (library with and without timing, 7 rounds, medians): 16k/256 +0.4 µs, 16k/512 -2.5 µs (noise), 48k/1024 +1.6 µs on the per-pass median; on the fastest pass +0.1 / +0.1 / -0.6 µs. The spread between rounds (14-18, 27-36, 39-47 µs) is larger than any difference.
- A target whose libc has no `CLOCK_MONOTONIC` builds with `make PROFILE=0` or supplies `-DAEC_NOW_US=<fn>`. Turning the macro off through `EXTRA_CFLAGS=-DAEC_STAGE_TIMING=0` is a redefinition; use `PROFILE=0`.
- `test_aec_array_flow`: the check after a non-finite hop asserted that the record is zeroed, which `aec.h` does not say (a call refused before the hop starts leaves the record untouched) and which never ran with timing on (it fails at the previous commit with timing built in). It now compares the record before and after the refusal.

## [Unreleased] — 2026-11-16 — The partition power sums as a kernel

No output change: the four baseline streams compare with 0 mismatches at `SIMD=1` and `SIMD=0`, and `aec_wav` output on four clips (16 kHz / 256, 16 kHz / 512 twice, 48 kHz / 1024; levels 0 without the residual stage, 1 with `--cng`, 2) is byte-identical to the previous build at both settings.

The two per-hop sums over the partitions of `x2_cache` (shadow NLMS step size and Kalman update) go through `sk_row_sum_f32()` (`simd_aec.h`). The NEON twin keeps 16 bins of the sum in registers over the partitions, in partition order. 44 to 46 % less time per call at 129, 257 and 513 bins on this host; the change in the linear stage is below the spread of repeated runs on a loaded machine (about 1 to 5 % at most). At `SIMD=0` the scalar twin is the old loop and the time is unchanged.

## [Unreleased] — 2026-11-16 — The stationarity window sums and flags as one kernel

No output change: the four baseline streams compare with 0 mismatches at `SIMD=1` and `SIMD=0`, and `aec_wav` output on the same clips is byte-identical to the previous build at both settings.

`stationarity_estimator_update_stationarity_flags()` forms the window sums and the flags in one pass, `sk_stationarity_window_flags_u8()` (`simd_aec.h`); the NEON twin keeps each bin group's sum in registers over the rows (storage order, reverb last). 19 to 23 % less time per call at 129, 257 and 513 bins and 6 to 9 % less in the steering stage on this host; at `SIMD=0` the scalar twin keeps the 128-bin row-major blocks and the time is unchanged.

## [Unreleased] — 2026-11-16 — Comfort-noise draws as a kernel

No output change: the four baseline streams compare with 0 mismatches at `SIMD=1` and `SIMD=0`, and `aec_wav --cng` output on four clips (16 kHz / 256, 16 kHz / 512 twice, 48 kHz / 1024; levels 1 and 2) is byte-identical to the previous build at both settings.

`aec_cng_fill()` draws through `sk_cng_inject_f32()` (`simd_aec.h`). The NEON twin takes the generator steps of four bins at once from the jump-ahead form of the one-step recurrence, so the draw sequence and the stored generator state are the one-per-bin sequence for every starting value. 40 % less time per call at 129, 257 and 513 bins and 3 % less in the residual stage on this host; at `SIMD=0` the loop is the old one and takes 0 to 10 % longer.

## [Unreleased] — 2026-11-16 — `AEC_LAYOUT_VERSION` 5: smaller pools, one admission rule, `aec_level_floors()`

`aec_process()` output is unchanged (the four baseline streams compare with 0 mismatches at `SIMD=1` and, for the changes before the lane buffer sharing and the view split below, at `SIMD=0` too; those two touch no SIMD kernel; `split_entry` differs in its 21 `META` pool-size lines against the layout-4 streams). A stored descriptor from layout 4 is refused by `aec_init_ex()`; re-query `aec_get_mem_size()`. `sizeof(Aec)` 6,048 to 6,032 B; single-channel pools -528 / -1,040 / -2,064 B at 16k/256 / 16k/512 / 48k/1024 (239,232 / 351,088 / 774,000 B balanced `MATCHED`), array `STATE` / `ESTIMATE` / `FULL` (balanced `MATCHED`, `delay_num_filters = 5`) 420,736 / 514,688 / 524,448 B at 16k/256 (-140,720 / -142,304 / -142,544 B), 615,536 / 802,160 / 820,112 B at 16k/512 (-233,600 / -236,720 / -236,960 B) and 1,366,976 / 1,738,944 / 1,776,352 B at 48k/1024 (-527,024 / -533,216 / -533,456 B; the lanes are now 73 / 76 / 77 % of the `FULL` pool, were 86 / 89 / 90 %), `aec_res_get_mem_size()` -240 B; the array carve text gains the shared per-call region (`build_flags_hash` 3,368,782,680 to 72,260,157, LP64).

- The single-channel shifted-delay admission (Path B) and the array's use the one `AecDelayAdmission` of `aec_impl.h` (was `AecArrayAdmission`; `AEC_ARRAY_DELAY_CHANGE_*` are `AEC_DELAY_CHANGE_*`; no candidate is `candidate = -1`). `make test-delay-admission` pins its 32-sample band, 16-sample confirmation window and 3-hop life through the hook `aec_testing_track_delay_shift()`.
- `mean_sq()` uses `sk_sum_sq_pairwise_f32()` (the same tree over the same products), so `Aec::scr_sq` is gone; `AecRes` rebuilds from `sg.cfg` / `sg.tun` instead of a second copy (`sg_cfg`, `sg_tun`, 236 B).
- `aec_level_floors(level, AecLevelFloors*)` replaces `aec_level_floor_db()`, `aec_level_dt_floor_db()` and `aec_level_split_floor_enabled()` (`aec_internal.h`). `aec_unusable()`, `aec_now_us()`, `aec_ring_*`, `AecFarShare`, the `aec_duty_*` API and the admission moved from `aec_array_internal.h` to `aec_impl.h`.

- **Array lanes borrow their per-call buffers.** The four lanes run one after another and each finishes its linear and residual stage before the next starts, so a buffer a lane writes before it reads inside one call is carved once per array (`AecLaneScratch`, `aec_lane_scratch_get_mem_size()` / `aec_lane_scratch_bind()`, 45,360 / 74,784 / 169,520 B for `ESTIMATE` / `FULL`, counted in `fusion_bytes`) instead of four times: the main and shadow filters' transform rows and constant window tables (`PbfdafScratch`, `pbfdkf_init_static_borrowed()` / `pbfdaf_init_static_follower_borrowed()`), the post-chain magnitude / PSD / mask rows, the FORM and filter-analyzer scratch, the tap span and the hop outputs no lane reads. What the donor lane hands to lanes 1-3 (`far_psd`, the scaled far block, the narrow-band mask), what a lane keeps from one hop to the next and what the views return stay per lane. A lane pool is 95,808 / 156,544 / 342,864 B (was 142,192 / 233,376 / 516,480; `STATE` 74,016 / 113,248 / 256,560 B). Single-channel pools do not move. The four baseline streams (`aec_array`, `aec_array2`, `split_entry`, `split_entry2`, `SIMD=1`) compare with 0 mismatches against streams captured before the change, `META` lines included, and again with the scratch overwritten with garbage before every lane call.
- **`AecLinearView` / `AecResidualView` hold what a caller outside the library reads.** Removed from `aec.h`: `AecResidualView.res_floor_protect` and `AecLinearView.n_freqs`, `echo_spec`, `near_spec`, `far_spec`, `generation`, `erle_factor`, `divergence`, `erl_estimate`, `shadow_dt`, `is_stationary_dt`, `filter_once_converged`, `epc_active` (nothing in Audio_ALG, the board samples, `aec_wav` or the documented examples reads them). The complete views are `AecLinearViewFull` / `AecResidualViewFull` with `aec_get_linear_full()` / `aec_get_residual_full()` in `include/internal/aec_view_internal.h`; the array flow and the AEC tests read those, and the public getters copy the public fields out of them (`make test-lifecycle` compares the two on a single channel and on an array). A field that only an Audio_ALG test reads stays public. `struct_size` is unchanged in kind; `sizeof(AecLinearView)` 120 to 64 B; `sizeof(AecResidualView)` stays 136 B (the removed `int` was padding), LP64. `AecLinearView` and `AecResidualView` changed layout and `struct_size` cannot detect it (a caller built against the layout-4 header passes a larger size and is accepted): rebuild every consumer, board samples included, against this `aec.h`.

## [Unreleased] — 2026-11-16 — aec.h declaration comments cut to three lines

Comments only. Nine long declaration comments in `aec.h` (refused hop, array instance, reset serial, synthesis window, residual view, linear-view seam, create / destroy / reset, external realign, array diagnostics) are three lines each; the detail is in `docs/reference/lifecycle.md` and `integration.md`.

## [Unreleased] — 2026-11-16 — bench_rtf and parity_aec_e2e check the aec_process return

Test drivers only. `bench_rtf` warns with the count of refused hops and stops on any other error; `parity_aec_e2e` stops on any non-`AEC_OK` return.

## [Unreleased] — 2026-11-16 — Test helpers: one all-zero check and an explicit n_mics table

Test only. `aec_pool_test_util.h` has one `aec_test_all_zero_f32()` for the refusal and linear-context tests, and the fresh-view scenario iterates an explicit `n_mics` table.

## [Unreleased] — 2026-11-16 — test_dsp_refusal: the whole-hop poison sample exists once

Test only. One `whole_hop_sample()` helper serves the single-channel and array rows, and the array poison table has a whole-hop flag.

## [Unreleased] — 2026-11-16 — The residual stage's two exclusive branches are an if / else

No output change. `aec_stage_residual()` chooses between the residual block and the linear-only steering with `if` / `else`, and `aec_steer_mu()` no longer takes a `clear_per_bin` flag.

## [Unreleased] — 2026-11-16 — A redundant alias and two misplaced comments in the hop functions removed

No output change. Cleanup in `aec_prepare_post_dt()` and `aec_run_post_chain()`.

## [Unreleased] — 2026-11-16 — Whole-body bare blocks and odd indentation left from the function split removed

No output change. The five carve functions lose their whole-body bare blocks and `aec_align_reference()` its odd indentation.

## [Unreleased] — 2026-11-16 — The post chain reuses the hop's sums of squares

No output change. `HopPowers` carries the hop's raw sums of squares of the microphone and the linear residual into `aec_post_run()` (`AecPostRunIn.mic_sum_sq`, `err_sum_sq`), which no longer sums the same two hops again.

## [Unreleased] — 2026-11-16 — The render base is chosen in one place

No output change. `aec_post_render_base()` (inline, `aec_post.h`) holds the rule "the output guard's base while it is active, else the selected error" that `aec_render_base()` and `aec_post_run()` each spelled out.

## [Unreleased] — 2026-11-16 — Refusal timing, the whole-hop threshold, the output below FULL and the array reset serial documented

Text only (`aec.h`, `lifecycle.md`, `integration.md`, `c_user_manual_zh_TW.md`, `aec_c_api.html`).

- A reference sample is refused on the hop whose aligned reference contains it: the hop that carried it when it reaches the filters at once (`EXTERNAL_ALIGNED`, a delay not yet locked, or a delay plus the sample's offset in the hop below `hop_size`), otherwise `floor((delay + offset) / hop_size)` hops later. With residual `OFF`, which computes no reference power, a sample near 2e19 is refused two to three hops after that, and one of 1e20 or more at once.
- A whole microphone hop is refused once `hop_size * rms^2` exceeds 3.4e38 (rms 1.6e18 at hop 128, 1.2e18 at 256, 8.2e17 at 512), in every residual mode.
- Below `FULL`, `aec_process()`'s `out` is `AecLinearView.linear_hop` (before FORM selection), not `formed_hop`.
- `aec_post_reset_serial()` is for inequality comparison only; an array instance moves it by four per reset event.

## [Unreleased] — 2026-11-16 — A hop whose energy detector or reference power overflows is refused in every mode

**Behaviour change (integrators read this).** The closed-hop check now also requires `dt_from_energy` finite in every residual mode and the hop's reference power finite when a residual stage runs. A whole microphone hop whose sum of squares overflows is refused by residual `OFF` too (before: accepted, leaving `dt_from_energy` NaN until the next reset), and one reference sample near 2e19 is refused on the hop that carries it in `STATE` (before: accepted with `far_power` inf and refused hops later). Residual `OFF` computes no reference power, so a reference sample near 2e19 is refused there two to three hops after it reaches the filters.

## [Unreleased] — 2026-11-16 — A refused single-channel hop zeroes aligned_far_hop as well

After `AEC_ERR_DSP` a single channel's `aec_get_linear()` returned the refused hop's reference in `aligned_far_hop`, although the documented view reads zeros. `aec_refuse_hop()` now zeroes it (an array's was already zero).

## [Unreleased] — 2026-11-16 — aec.c: scaffolding left from the function split removed

No output change. Leftovers of the stage split in `aec.c` removed: blank lines before closing braces, whole-body bare blocks and `if` wrappers, two empty `if` branches, underscore-prefixed locals and an unused `(n)` argument of the carve slice macros.

## [Unreleased] — 2026-11-16 — Test and example helpers shared instead of repeated

No behaviour change. `aec_cli_parse_level()` calls `aec_cli_parse_int()`, `test/baseline2_signal.h` shares the LCG step, modulation and hash with `split_signal.h`.

## [Unreleased] — 2026-11-16 — The array shares the single channel's reference-ring sizing; smaller view and counter helpers

No output change. The array sizes its reference ring through `aec_ref_ring_samples()` like the single channel (the array gains its overflow guard), `aec_far_fft_real_compute_count()` uses the saturating-add helper, and `aec_get_residual()` reads its three scalars directly instead of filling a linear view. Docs: the function map (new C-only row `aec_render_base`).

## [Unreleased] — 2026-11-16 — The hop's mean squares are computed once

No output change. A hop's mean squares are computed once and passed on (`aec_hop_powers()`), and the converged-mu-freeze-or-simple-ratio rule exists once (`aec_steer_mu()`, formerly `aec_update_post_mu()`). `AecPostRunIn.shared_far_active` and `AecPost::far_active` are replaced by `AecPostRunIn.far_mean_sq`. Docs: the function map and the two HTML sequence pages.

## [Unreleased] — 2026-11-16 — A fresh instance's linear view reads zeros

`aec_get_linear()` before the first hop returned pointers to pool or heap bytes the carve never cleared. `aec_carve_hop_scratch()` now zeroes `raw_output` and `far_hop` at init (single channel and array lane 0). Processing is unchanged. `aec_reset()` still leaves the last hop's views readable until the next hop; only a refused hop zeroes them.

## [Unreleased] — 2026-11-16 — aec_build_sg_config() zeroes the tuning it hands out

`aec_build_sg_config()` zeroes its `SuppressionGainTuning` before filling it, so the four ENR/SU table pointers its comment promises to be NULL are NULL. No output change.

## [Unreleased] — 2026-11-16 — A hop whose detector state overflows is refused

**Behaviour change (integrators read this).** A hop of large finite microphone samples whose sum of squares overflows `float` (`hop_size * rms^2` above 3.4e38) was accepted although `dt_indicator` and the instantaneous ERLE stayed NaN until `aec_reset()`. `aec_closed_finite()` now also requires `inst_erle_smooth` and `last_dt_indicator` to be finite: such a hop is `AEC_ERR_DSP` with the instance reset, for a single channel and for an array lane. Normal input is unchanged.

## [Unreleased] — 2026-11-16 — A refused array hop leaves the linear view reading zeros, as a single channel

After an array `AEC_ERR_DSP`, `aec_get_linear()` (lane 0's view with the shared alignment) still answered `AEC_OK` with the refused hop's lane-0 buffers, which the reset does not clear; with `3e38` on microphone 0 they were not finite. `aec_refuse_hop()` now zeroes that lane's `linear_hop` and spectra as it does a single channel's (`aec_array_impl_linear_lane()`); `aligned_far_hop` was already zero. Output of accepted hops unchanged (the four baseline streams byte-identical, `SIMD=1` / `0`).

- `test/test_dsp_refusal.c`: each of the 108 array rows also asserts `aec_get_array()` is `AEC_ERR_SEQUENCE` and the linear view reads zeros after the refusal; with the zeroing removed all 108 fail.
- Docs: `aec.h` (array refused hop), `aec_array_internal.h`, `docs/reference/lifecycle.md`, the function map (`aec_refuse_hop`, new row `aec_array_impl_linear_lane`).

## [Unreleased] — 2026-11-16 — Internal `AecTuning.max_delay_ms` deleted

**Nothing reads it any more.** Since the array's shared ring follows the estimator reach (entry below), `max_delay_ms` was read only by the array's `<= 4096` config check. The field, its default (1024 ms), its `[0, 60000]` range check and the array check are deleted. `sizeof(AecTuning)` 168 to 164 B (`sizeof(AecResolved)` 216 to 212 B); `sizeof(Aec)` stays 6,048 B (padding), every pool (single channel and array, every grid, residual mode, delay mode and `delay_num_filters`) is unchanged, and the `split_entry` `META` lines (pool sizes, `sizeof_Aec`) are identical before and after. `AEC_LAYOUT_VERSION` stays 4 and `build_flags_hash` is unchanged. The four baseline streams are byte-identical to the streams captured before this change (`SIMD=1` and `SIMD=0`, `META` included) and the reach boundary band is identical.

- Tests removed with the field: the two short-ring scenarios (`test_aec_array.c` `test_estimate_beyond_ring`, `test_aec_array_flow.c` `test_unservable_delay_is_not_accepted`; with the reach-sized ring an estimate past the ring cannot occur), the array's `max_delay_ms` range rows (`test_aec_array.c`), the `FIXED` "max_delay_ms inert" half of `test_linear_context.c` and the `AfConfig.max_delay_ms` plumbing of `test_aec_array_flow.c`.
- `test_config_validation.c`: the default-tuning hashes are re-pinned (FNV-1a over the `AecTuning` bytes). The new pins equal the old struct's hashes with the four `max_delay_ms` bytes skipped, computed from the struct before this change, so every other default is unchanged.
- Python is unchanged (`python/modules/config.py` keeps `max_delay_ms` and `delay_buffer_ms`; gaps D-16, D-17). Docs: `aec_params.html` (row deleted), `aec_params_internal.html` (delay group 5 to 4 fields, total 42 to 41), `aec_methods.md`, `aec_array_internal.h`, the function map (`aec_array_impl_config_ok`, D-16, D-17).

## [Unreleased] — 2026-11-16 — An array lane whose results overflow is refused like a single channel

**Behaviour change (integrators read this).** The array instance checked only its fused spectra: with `STATE` (no fusion) nothing refused an overflowing hop, and a microphone sample that overflowed one lane's power trackers but not the fusion (for example `1e20`) was accepted and left that lane unconverged until `aec_reset()`; with `STATE` and `3e38` the lane views it handed out were not finite. Now `aec_process_linear()` checks each lane as a single channel does (the lane's linear view after its linear stage; its power trackers and residual view after its residual stage) in every residual mode. A lane that fails makes `aec_process_linear()` return `AEC_ERR_DSP` with the whole instance reset as by `aec_reset()`; no hop is open (`aec_process_residual()` is `AEC_ERR_SEQUENCE`, `AEC_ERR_MODE` with `STATE`), and from the next hop the instance equals a fresh one. With `ESTIMATE` / `FULL` a refusal that the fusion used to make in `aec_process_residual()` now comes one call earlier, from `aec_process_linear()`. Measured thresholds (one sample, the test scene): a microphone sample from about 1.9e19 on any lane, the reference from about 3.8e19 to 5.6e19 with `STATE` and 1.1e15 to 1.4e15 with `ESTIMATE` / `FULL`, every grid; the same as a single channel's microphone threshold.

**Normal input unchanged.** The checks only read; the four baseline streams (`SIMD=1` / `0`) are byte-identical to the streams captured before this change, the Audio_ALG `pipelines/4ch` tests and `pyc_parity_pipelines.sh` give the same results. Python has no such refusal (C-only guard, as for the single channel).

**Cost.** Per hop the array runs the single-channel split-pair checks four times: from the single-channel figures of the entry below, about 280 / 580 / 1,110 ns at `SIMD=1` (16k/256, 16k/512, 48k/1024; about 1 % of a 25 to 90 us array hop) and 2.0 / 4.0 / 7.2 us at `SIMD=0`. An end-to-end A/B of the array hop (min of nine 600-hop runs, `STATE` and `FULL`) on this shared machine stays inside its run-to-run spread. No pool or struct changes.

- `aec_lane_process_linear()` / `aec_lane_process_residual()` (`aec.c`) answer `AEC_ERR_DSP` from `aec_linear_finite()` / `aec_closed_finite()`; `aec_array_impl_process_linear()` already turns a lane error into `aec_refuse_hop()`.
- `test/test_dsp_refusal.c`: 108 array rows (three grids, `STATE` / `ESTIMATE` / `FULL`, each of the four lanes, microphone `3e38` / `1e25` / `1e20`) assert the refusal by `aec_process_linear()` on the poisoned hop, the serial, no open hop, and fresh-instance equality of every later hop (aligned far, lane hops and spectra, fused views, gained spectrum); 297 sweep rows (lane 0, lane 3 and the reference, eleven values from `1e10` to `FLT_MAX`) assert finite results on every accepted hop and a refusal from `1e20`; 27 control rows (silence, full-scale square, full-scale DC) assert no refusal. Against the library without this change 267 checks fail; removing the lane closing check, or the array's use of its result, fails 144; removing the lane linear check alone passes (the closing check of the same call also catches every such hop: kept for the single channel's order of checks).
- Docs: `aec.h` (array errors paragraph, "Refused hop"), `aec_array_internal.h`, `c_impl/README.md`, `docs/c_user_manual_zh_TW.md`, `docs/reference/{lifecycle,integration}.md`, `docs/html/reference/{aec_c_api,aec_sequence}.html`, `docs/c_python_function_map.md` (`aec_lane_process_*`, `aec_linear_finite`, `aec_closed_finite`, `aec_array_impl_process_linear`).

## [Unreleased] — 2026-11-16 — The array's shared reference ring is sized like a single channel's (estimator reach + hop)

**Pool change, output unchanged.** An array instance with `delay_mode = MATCHED` sized its shared aligned-far ring from the internal `max_delay_ms` (`ceil(1024 ms * sample_rate / 1000) + 2 * hop + 1` samples). It now uses the single-channel rule, `delay_aec_max_delay_samples(rate, delay_num_filters) + hop` (`4 * ((n - 1) * 384 + 501) + hop` samples, three times the first term at 48 kHz): the ring holds every delay the estimator can report and nothing beyond it. Array `MATCHED` pools (balanced, `delay_num_filters = 5`, LP64) `STATE` / `ESTIMATE` / `FULL`: 16k/256 594,928 / 690,464 / 700,464 to 561,456 / 656,992 / 666,992 B (-33,472), 16k/512 883,120 / 1,072,864 / 1,091,056 to 849,136 / 1,038,880 / 1,057,072 B (-33,984), 48k/1024 1,994,896 / 2,373,056 / 2,410,704 to 1,894,000 / 2,272,160 / 2,309,808 B (-100,896); `ring_bytes` 66,576 / 67,600 / 200,720 to 33,104 / 33,616 / 99,824 B. With fewer filters the ring shrinks with the reach (16k/256 `FULL`, n = 1: 621,936 B). `FIXED`, `EXTERNAL_ALIGNED`, single-channel pools, `sizeof(Aec)`, `AEC_LAYOUT_VERSION` (4) and `build_flags_hash` are unchanged.

**Evidence.** The four baseline streams (`split_entry`, `split_entry2`, `aec_array`, `aec_array2`; `SIMD=1` and `SIMD=0`) are byte-identical to the streams captured before this change except the `META` lines; the Audio_ALG `pipelines/4ch` tests (`SIMD=1` / `0`) and `pyc_parity_pipelines.sh` (`results.json`) give the same results; the reach boundary band (a `STATE` array, true delays from reach - 100 to twice the reach, n = 5 and 1, three grids: 36 cases) is identical hop for hop. The checks that refuse an estimate past the ring (`update_shared_delay()`, `align_render()`) stay as a defence; with this ring they can no longer fire.

- The internal `max_delay_ms` now sizes nothing; the array still range-checks it (<= 4096).
- Python: `python/` has no array instance and is unchanged; the array's Python twin (Audio_ALG `pipelines/4ch/pipeline.py`) still sizes its ring from `max_delay_ms` and agrees at its defaults. Function map: `array_dims` is `partial`, gap D-17.
- `test/test_aec_array_flow.c` (pool versus bank size): the ring now grows with `delay_num_filters` by one filter's reach (1,536 samples, 6,144 B at 16 kHz); the flow buffers stay fixed and every bank step still costs the same (11,264 B at 16k/512: estimator 5,120 + ring 6,144).
- `test/test_memory_layout.c` array pins, `c_impl/STATIC_MEMORY.md`, `aec.h` (array differences paragraph), `aec_tuning.h`, `docs/reference/{lifecycle,configuration,integration,linear_frontend}.md`, `docs/c_user_manual_zh_TW.md`, `docs/aec_methods.md` and `docs/html/reference/{aec_c_api,aec_params,aec_params_internal,aec_usage}.html` follow (the lane share of the `FULL` pool is now 86 / 89 / 90 %); the `FIXED` / `EXTERNAL_ALIGNED` array figures quoted there, already 816 B stale, are re-measured.

## [Unreleased] — 2026-11-15 — A single-channel hop whose results overflow is refused with `AEC_ERR_DSP` and the instance reset

**Behaviour change (integrators read this).** Before, one finite input sample large enough to overflow the arithmetic (for example `3e38` on the microphone, or `1e20` on the reference) left a single-channel instance non-finite, or permanently unconverged with finite output, until the caller reset it, while an array instance answered `AEC_ERR_DSP`. Now `aec_process_linear()` checks the `AecLinearView` vectors it hands out, and the call that closes the hop (`aec_process()`, `aec_process_residual()`, `aec_process_linear()` with `AEC_RESIDUAL_OFF`; also `aec_process_linear_shared_far_from()`) checks the near / linear-output power trackers and the `AecResidualView` vectors as far as `AecConfig.residual` reaches. When one is not finite the call returns `AEC_ERR_DSP` after resetting the instance as `aec_reset()` does: `aec_process()` writes `hop_size` zeros to `out`, `aec_post_reset_serial()` moves, no hop is open, the residual view and `aec_apply_gain()` / `aec_apply_cng()` answer `AEC_ERR_SEQUENCE`, and the linear view reads zeros. From the next hop the instance is bit-identical to a fresh one. Measured thresholds (16 kHz and 48 kHz, every residual mode): microphone from about 1.9e19, reference from about 1.3e19, or 5.6e14 with `ESTIMATE` / `FULL`; full scale is 1.0. A caller that treated every non-`AEC_OK` return as "reset and output silence" needs no change; one that ignored the return must now treat the hop as silence and clear its own synthesis (the contract is in `aec.h`, "Refused hop").

**Normal input unchanged.** The four baseline streams compare with 0 mismatches, `META` lines included, against streams captured before this change (`SIMD=1` and `SIMD=0`); `test-pyc-parity` and `test-parity-e2e` pass unchanged. The Python twin has no such refusal: the check acts only on overflowing input, so default-path parity is unaffected (function map rows `aec_refuse_hop`, `aec_linear_finite`, `aec_closed_finite`).

**One shared exit.** The array's reset-and-return (`array_dsp_error`) and its complex finite check (`complex_vector_finite`) are now `aec_refuse_hop()` and `aec_cvec_finite()` in `aec.c` (declared in `aec_array_internal.h`, built with `AEC_NO_ARRAY=1` too), used by both instance kinds. The array's checks are unchanged: it still checks only the fused spectra, so with `STATE` (no fusion) or a microphone sample that overflows a lane's trackers but not the fusion (measured: `1e20` at 16 kHz/256) it accepts the hop; `aec.h` now says so.

**Cost (this machine, the check alone, per hop).** `aec_process()`: 16k/256 39 ns, 16k/512 80 ns, 48k/1024 160 ns at `SIMD=1` (0.3 to 0.4 % of the hop), 290 / 550 / 1,060 ns at `SIMD=0` (0.7 to 1.2 %); the split pair adds the linear check: 70 / 145 / 277 ns (0.5 to 0.8 %) and 490 / 1,000 / 1,800 ns (1.2 to 2.2 %). End to end the A/B difference is inside the run-to-run spread. No pool or struct changes.

- `test/test_dsp_refusal.c` (`make test-dsp-refusal`, in `make test`): 104 poisoned rows (three grids, four residual modes, `aec_process()` and the split pair with four poisons; shared far at 16k/256 with the two microphone poisons) assert the refusing call and hop, `out`, the serial, the views and fresh-instance equality afterwards; 264 sweep rows assert that every accepted hop leaves finite views, `out` and trackers; 72 control rows (digital silence, a full-scale square wave on both inputs, full-scale DC; every mode, `aec_process()` and the split pair) assert that legal extremes are never refused; NaN / Inf input stays `AEC_ERR_ARG`; one array row. Mutation: each check group, the reset, the view zeroing, the `out` zeroing and each entry point's call site fail the test when removed, and a refusal of loud legal input fails the control rows (16 of 21 mutants killed). Single vectors inside a group (the linear view's echo / near / far spectra, the comfort-noise PSD, the near-power tracker) do not fail it alone: on the tested inputs another vector of the same group overflows on the same hop.
- Docs: `aec.h` (`AecStatus`, "Refused hop", `aec_post_reset_serial()`, the array's error paragraph), `aec_internal.h`, `docs/c_user_manual_zh_TW.md`, `docs/reference/lifecycle.md`, `docs/html/reference/{aec_c_api,aec_sequence,aec_usage}.html`, `c_impl/README.md`, `docs/c_python_function_map.md`.

## [Unreleased] — 2026-11-15 — `AecResidualView.delay_changed` / `delay_locked` removed, `AEC_LAYOUT_VERSION` 4

**Output unchanged.** The four baseline streams (`split_entry`, `split_entry2`, `aec_array`, `aec_array2`) compare with 0 mismatches, `META` lines included, against streams captured before this change at 16k/256, 16k/512 and 48k/1024, `SIMD=1` and `SIMD=0`. (`split_entry3` was folded into `split_entry2` earlier, and the pre-split golden files under `test/golden/` are no longer in the tree.)

**Removed from `aec.h`.** `AecResidualView.delay_changed` and `AecResidualView.delay_locked`. They repeated what two other views already carry, hop for hop: a single channel reads `aec_get_linear()` `delay_state` (`delay_changed` was `delay_state == AEC_LINEAR_DELAY_CHANGED`, `delay_locked` was `delay_state != AEC_LINEAR_DELAY_UNLOCKED`), an array reads `aec_get_array()` `delay.changed` / `delay.solid`. No code outside the tests read them. `sizeof(AecResidualView)` 144 to 136 B (LP64), 96 to 88 B (ILP32); the view is `struct_size`-checked, but a caller built against the old header passes the larger size, is accepted and reads `guard_lanes` / `far_hop` at the old offsets: rebuild every consumer.

**Pool layout.** The array's fused-frame record loses the same two fields, so every array pool is 16 B smaller (LP64): 16k/256 `STATE` / `ESTIMATE` / `FULL` 594,944 / 690,480 / 700,480 to 594,928 / 690,464 / 700,464 B, 16k/512 883,136 / 1,072,880 / 1,091,072 to 883,120 / 1,072,864 / 1,091,056 B, 48k/1024 1,994,912 / 2,373,072 / 2,410,720 to 1,994,896 / 2,373,056 / 2,410,704 B; `fusion_bytes` -16 B in every array configuration. Single-channel pools are unchanged. On ILP32 (clang `armv7`, `sizeof` only) the array control block goes from 1,932 to 1,924 B and the fused-frame record from 68 to 60 B; both control-block sizes round to 1,936 B in the 16-B carve, so the 32-bit array pools are expected not to move (not measured end to end). `AEC_LAYOUT_VERSION` 3 to 4 (a stored descriptor from layout 3 is refused by `aec_init_ex()`); the carve text and `build_flags_hash` are unchanged.

- Tests: the `frame_flags` hash of `test_aec_array.c` and the frame hash of `test/baseline2_array.h` take the same values from `AecArrayView.delay` (`changed`, `solid`), so the streams do not move; the equality checks between the removed fields and their duplicates (`test_aec_array.c`, `test_aec_array_flow.c`, `test_split_entry2.c`) are gone with them. Mutation: hashing a constant in place of `delay.solid` gives 52,046 `aec_array` and 3,204 `aec_array2` mismatches.
- `test/test_memory_layout.c` array pins and `AEC_LAYOUT_VERSION_PIN`, `c_impl/STATIC_MEMORY.md`, `docs/reference/lifecycle.md`, `docs/reference/integration.md`, `docs/sdk_migration_output_guard.md` and the HTML references follow.

## [Unreleased] — 2026-11-15 — No `_Static_assert` or `static_assert` remains; the tests pin those facts at run time

**Library output and object code unchanged.** Every `.o` of `libaec.a` and of `aec_wav` (`SIMD=1` and `SIMD=0`) is byte-identical to the objects built before this change (a static assert emits no code), and the four baseline streams compare with 0 mismatches against streams captured before this change. `AEC_LAYOUT_VERSION` stays 3.

- Removed: `_Static_assert(sizeof(AecMemReq) == 32)` in `include/aec.h`; four `_Static_assert`s in `include/internal/aec_impl.h` tying `AEC_DELAY_NUM_FILTERS_MAX`, `AEC_DELAY_FILTER_SIZE`, `AEC_DELAY_FILTER_INTRA_SHIFT` and `AEC_DELAY_DOWN_SAMPLING_FACTOR` to `DA_NUM_FILTERS`, `DA_FILTER_SIZE`, `DA_FILTER_INTRA_SHIFT` and `DA_DOWN_SAMPLING_FACTOR`; `_Static_assert(_Alignof(struct Aec) <= 16)` (the only C11-only construct in the headers); the two `static_assert`s of `test/test_header_cpp.cpp`.
- `test/test_memory_layout.c` has a new `check_header_facts()`: `sizeof(AecMemReq) == 32` and `offsetof(AecMemReq, bytes) == 24`, the four `AEC_DELAY_*` / `DA_*` equalities, and the alignment of `struct Aec` (C99 form: the member offset in a struct with a leading `char`), each failing with a message that names the fact. The `sizeof(Aec)`, pool-size, `AEC_LAYOUT_VERSION` and carve-hash pins that were already there are unchanged.
- `test/test_header_cpp.cpp` is also built with `-DAEC_HEADER_CPP_RUN` and run by `make test-header-cpp` (no library archive needed): the `AEC_LEVEL_*` digits 0..3, the delay-geometry names, `AEC_CNG_SEED == 42u` and the `AEC_PSD_SCALE_F * AEC_PSD_SCALE_INV_F == 1` identity are checked at run time. The compile-only pass over `aec.h` as C and as C++11 stays.
- A change to any of these facts no longer stops the compiler; it fails `make test-memory-layout` or `make test-header-cpp` (and so `make test` and `make test-full`). A build that skips the tests does not see it.

## [Unreleased] — 2026-11-14 — Dead fields removed, `AEC_LAYOUT_VERSION` 3, `AEC_OUTPUT_GUARD_API_VERSION` and two `AecArrayView` fields removed

**`aec_process()` output unchanged.** The four baseline streams (`split_entry`, `split_entry2`, `aec_array`, `aec_array2`) written from this tree compare with 0 mismatches against streams captured before this change at 16k/256, 16k/512 and 48k/1024, `SIMD=1` and `SIMD=0`. `split_entry` differs in exactly the `META sizeof_Aec` and `META mem_size_*` lines (6 per grid and build, 18 per stream file); `split_entry2`, `aec_array` and `aec_array2` are byte-identical.

**Pool layout (integrators read this).** `AEC_LAYOUT_VERSION` 2 to 3: a pool descriptor stored from a build with layout 2 is refused by `aec_init_ex()`, and `aec_init()` / `aec_create()` size the pool from the current header. Re-query `aec_get_mem_size()`; do not carry byte counts forward. LP64 sizes, before to after:

| | before | after |
|---|---:|---:|
| `sizeof(Aec)` (the pool header, 16 B aligned) | 6200 (6208) | 6048 (6048) |
| single channel 8k/256 | 211,552 B | 211,392 B |
| single channel 16k/256 | 239,920 B | 239,760 B |
| single channel 16k/512 | 352,288 B | 352,128 B |
| single channel 48k/1024 | 776,224 B | 776,064 B |
| array 16k/256 `STATE` / `ESTIMATE` / `FULL` | 595,744 / 691,280 / 701,280 B | 594,944 / 690,480 / 700,480 B |
| array 16k/512 | 883,936 / 1,073,680 / 1,091,872 B | 883,136 / 1,072,880 / 1,091,072 B |
| array 48k/1024 | 1,995,712 / 2,373,872 / 2,411,520 B | 1,994,912 / 2,373,072 / 2,410,720 B |
| array lane pool 16k/256 (`ESTIMATE`, `FULL`) | 142,880 B | 142,720 B |

Every pool shrinks by 160 B per `Aec` header (a single-channel instance has one, an array has one plus four lanes: 800 B); all deltas between configurations are unchanged. `aec_res_get_mem_size()` is unchanged. `libaec.a` `__TEXT` (host, summed `size`): `SIMD=1` 160,230 to 160,122 B, `SIMD=0` 140,774 to 140,678 B, `AEC_NO_ARRAY=1` 139,318 to 139,234 B and 122,990 to 122,914 B. `test/test_memory_layout.c` pins and the byte counts in `c_impl/STATIC_MEMORY.md`, `c_impl/README.md`, `docs/reference/lifecycle.md`, `docs/reference/wav_cli.md` and the HTML references are re-measured.

**Removed from `aec.h`.**
- `AEC_OUTPUT_GUARD_API_VERSION`. A build that guards on it (`#if !defined(AEC_OUTPUT_GUARD_API_VERSION) || ... < 2` / `#error`) now stops at the `#error`: use `#if !defined(AEC_CNG_NR_GAIN_FLOOR)` (see `docs/sdk_migration_output_guard.md` section 2). The output-guard API itself (`base_spec`, `aec_apply_gain()`, `aec_apply_cng()`, `aec_post_reset_serial()`, `aec_synthesis_window()`) is unchanged.
- `AecArrayView.n_channels` and `AecArrayView.hop_size` (176 B instead of 184 B on LP64; `aec_array_linear_aec_count()` and `aec_hop_size()` return the same values). `AecArrayView` has no version constant: a caller built against the old header and run against the new library is accepted and reads shifted fields, so rebuild every consumer (board samples included) against this `aec.h`.
- Kept after review, because an integration document names them as outputs (no code outside this repository reads them): `AecResidualView.delay_changed` / `delay_locked` (`reference/integration.md`, Audio_ALG `pipelines/reference/call-flow.md`), `AecLinearView.echo_spec` / `near_spec` / `far_spec` / `epc_active` / `filter_once_converged` / `is_stationary_dt` / `shadow_dt` (`reference/integration.md`, `sdk_migration_output_guard.md`).

**Removed from internal structures** (all written, none read): `Aec.fft_size` (the `fft` parameter of `aec_init_config_state()` goes with it; tests read `main_filter.base.fft_size`), `AecState.st`, `FilterDelayEstimate.delay`, `ResidualEchoEstimator.hop_size` / `reverb_decay` / `reverb_mild_decay_scale` / `sample_rate` (the `ree_init()` parameters stay: they feed the per-hop decay and the noise-floor growth), `SuppressionGainConfig.use_wallclock_ema_alpha` (the C power IIR was always the literal 0.9 / 0.1; the Python `use_wallclock_ema_alpha` keyword stays) and five members of `AecBalancedRateDims` (`block_size`, `filter_length`, `filter_taps_size`, `n_bins`, `n_partitions`; `python/diag/gen_aec_balanced_config_h.py` emits the table, `include/internal/aec_balanced_config.h` is regenerated, `test/aec_ref_tables.h` is byte-identical; the 4-row table is 96 B smaller). 17 `AECB_*` macros are now definition-only in the generated header.

**Docs.** `docs/c_python_function_map.md` rewords the `external_delay_samples` rows (C keeps only the `reported` flag; the shared `FilterDelay` row is split in two, so the Python-only statistics read `py_only_field` 58 and `py_only_none` 137; the C-function counts are unchanged), `docs/timing_constant_inventory.*` drops the pointer to the removed C field, `docs/sdk_migration_output_guard.md` gains the layout 3 and `AecArrayView` rows.


## [Unreleased] — 2026-11-14 — `aec.c` split into three files, test and generator clean-up, `aec_wav --no-delay-est` removed

**`aec_process()` output unchanged.** The four baseline streams (`split_entry`, `split_entry2`, `aec_array`, `aec_array2`) compare with 0 mismatches against streams captured before this change, at 16k/256, 16k/512 and 48k/1024, `SIMD=1` and `SIMD=0` (and `AEC_NO_ARRAY=1` for the single-channel two). `aec_wav` writes the same samples and the same stderr as before on the 17 accepted runs (and refuses the 18th, `--fft-size 512` at 48 kHz, as before). `sizeof(Aec)` and every pool size are unchanged.

**`aec_wav --no-delay-est` is removed** (use `--delay-mode fixed|external`, which `test-cli-delay-flags` covers). The Python orchestrator keeps its own `--no-delay-est`.

**`aec.c` is split** (code moved unchanged; the preprocessed text of every moved function is identical except the spelling of `INFINITY` in `aec_cng_fill()`):
- `src/aec_tuning.c` also holds the level table and its lookups (`aec_level_name()`, `aec_level_floor_db()`, `aec_level_dt_floor_db()`, `aec_level_split_floor_enabled()`), the signal-grid table (`aec_resolve_signal_grid()`, `aec_is_valid_sample_rate()`, `aec_default_fft_size()`), `aec_config_from_level()`, `aec_resolve()` and `aec_resolve_tuned()`. `aec_resolved_valid()` stays in `aec.c` (it needs the array rules and the lane functions call it) and is now declared in `aec_tuning.h`.
- `src/aec_post.c` also holds `aec_apply_gain_spec()`, `aec_cng_fill()` and `aec_apply_cng_spec()`. `aec_apply_gain()` and `aec_apply_cng()` stay in `aec.c`.
- New `src/aec_config_build.c` holds the five definitions of `aec_config_build.h` (`aec_build_sg_config()`, `aec_sg_apply_level()`, `aec_sg_tables_bytes()`, `aec_sg_tables_decode()`, `aec_synth_window_build()`) and `aec_sg_tables_expand()`, now callable from `aec.c`. It is the only object that instantiates the generated rate table and the window half table; the rest of the library reads a rate row through the new `aec_balanced_rate_dims()`. `AECB_SQRT2_SIN_LUT` lives in `aec_post.o` only. Every `AECB_*` table is in exactly one object of `libaec.a`.
- `libaec.a` has one more member (`aec_config_build.o`) and three more internal symbols (`aec_balanced_rate_dims`, `aec_resolved_valid`, `aec_sg_tables_expand`); no symbol is removed and every `aec.h` function (including `aec_apply_gain_spec()`, `aec_cng_fill()` and `aec_apply_cng_spec()`, now in `aec_post.o`) is exported as before. The source list is part of `CFG_SIG`, so every configuration lands in a fresh `obj/` and `bin/` directory.
- Docs: `docs/c_python_function_map.md` re-keys 21 rows to the new files and adds `aec_balanced_rate_dims` (statistics recounted), `python/tests/test_signal_grid.py` reads `AEC_GRID_TABLE` from `src/aec_tuning.c`, and two HTML pages and `postfilter.md` name the new files. A harness that lists AEC sources by name (Audio_ALG `pipelines/tests/pyc_parity/pyc_parity.py` `SOURCES`) should add `aec_tuning.c`, `aec_post.c` and `aec_config_build.c`.

**Removed.**
- `AECB_ENABLE_8K`, `AECB_ENABLE_16K_256`, `AECB_ENABLE_48K` (compile-time table-trim switches, all defaulting to 1, never set to 0 anywhere): `python/diag/gen_aec_balanced_config_h.py` no longer emits them, `include/internal/aec_balanced_config.h` is regenerated without them and without the 512-point fallback window table that only the `AECB_ENABLE_48K=0` branch used. All 28 library objects are byte-identical to the previous ones.
- Twelve `python/diag` generators that no C test, Makefile target, script, document or other generator named: `gen_fft_golden.py`, `gen_filter_analyzer_golden.py`, `gen_filter_quality_golden.py`, `gen_fullband_erle_golden.py`, `gen_initial_state_golden.py`, `gen_render_signal_analyzer_golden.py`, `gen_residual_echo_estimator_golden.py`, `gen_reverb_frequency_response_golden.py`, `gen_reverb_golden.py`, `gen_saturation_detector_golden.py`, `gen_stationarity_estimator_golden.py`, `parity_aec3_scale_golden.py`. The generators named by `docs/timing_constant_inventory.md` or imported by another generator stay.
- The 23 `bench_*` micro-benchmarks, their `BENCH_REPS` driver and printing in `test/simd_selftest_aec.c`; every scalar-versus-NEON check stays (111,559,370 checks, the same count as before, at `SIMD=1` and `SIMD=0`).

**`test-split-entry-3` is folded into `test-split-entry-2`.** The 18 bad-relock scenes with their decision expectations and the check that all four outcomes (natural revert, kept with an estimate, kept without one, soft path) occur are now in `test/test_split_entry2.c`; `test/test_split_entry3.c` and the Makefile gate are gone (variables, run rule, link rule, the three goal lists, help text). The `split_entry3` baseline stream no longer exists: `split_entry2` is now the old `split_entry2` stream followed by the old `split_entry3` stream, byte for byte, so **baselines captured before this change must be re-captured** for `test-split-entry-2` (`split_entry`, `aec_array`, `aec_array2` are unaffected). A flipped expectation in the 48 kHz `rj_soft` scene makes the merged test fail.

## [Unreleased] — 2026-11-14 — Headers under `include/internal/`, per-hop trace removed, unused code removed

**`aec_process()` output unchanged.** The five baseline streams written from this tree (`split_entry`, `split_entry2`, `split_entry3`, `aec_array`, `aec_array2`) differ from the pre-round streams in exactly one place: the `META sizeof_Aec` line of `split_entry` (3 lines per build), `0x1840` to `0x1838`, i.e. `sizeof(Aec)` 6208 to 6200 B on the 64-bit host. The pool sizes are unchanged (the pool carves `ALIGN16(sizeof(Aec))`, 6208 B before and after; `aec_get_mem_size()` and the three pinned single-channel and array pool sizes are identical), `AEC_LAYOUT_VERSION` stays 2. `aec_wav` output is byte-identical at `SIMD=1` and `SIMD=0` on 17 runs (two files at 16 kHz and 48 kHz, levels 0 to 3, `--cng`, `--no-res`, `--fft-size 512`, fixed delay, `--delay-num-filters 2`) and its rejection of `--fft-size 512` at 48 kHz prints the same message.

**Header layout (integrators read this).** `include/` now holds only `aec.h`, the public header. `aec_internal.h`, `aec_balanced_config.h` and every header that was in `src/` (`aec_impl.h`, `aec_tuning.h`, `aec_post.h`, `delay_aec.h`, `pbfdkf.h`, ...) moved to `include/internal/`. An SDK Makefile that compiles `src/*.c` needs `-I include/internal` in addition to `-I include`; an integrator that uses only the public API (including `aec.h`) still needs just `-I include`. `aec_internal.h` is not public API and now sits beside the other internal headers, so a pipeline that includes it adds `-I <AEC>/c_impl/include/internal`. The `Makefile` adds both paths for the library, the CLI and the tests (`TEST_INC` is gone); `test-header-cpp` compiles `aec.h` with `include/` alone on the path. Preprocessed output of every `.c` under `src/`, `example/` and `test/` is identical before and after the move. `python/diag/gen_aec_balanced_config_h.py` writes `include/internal/aec_balanced_config.h`; `docs/c_python_function_map.md` and `docs/timing_constant_inventory.*` carry the new paths (the map's `pbfdkf.h` row counts in the header column now: `c_src_total` 521 to 520, `c_hdr_total` 4 to 5).

**Trace removed.** `include/aec_debug.h`, `src/aec_debug.c`, the per-hop trace block in `aec_process()` and `aec_wav --debug-trace` are deleted (`aec_debug_set_frame()`, `aec_debug_set_trace()`, `aec_debug_trace_active()`, `aec_debug_trace_row()`, `AecDebugTraceRow`). `aec_debug_status()` (in `aec.h`) and `aec_wav --debug` stay; a problem report attaches the `--debug` output instead of the CSV. The library has no stdio TU any more, so `NO_STDIO=1` selects no source and no flag (it only keys its own obj/bin directory) and `audit-no-stdio` is the regression gate. The versioning paragraph above no longer lists the `--debug-trace` CSV schema in the output contract.

**Trace scratch fields removed.** `AecPostTrace` (`aec_converged`, `gain_mean`) and the per-hop mean of the suppression gain are gone; `AecPost.far_active` replaces `trace.far_active`. `sizeof(Aec)` 6208 to 6200 (`AEC_SIZEOF_PIN` in `test/test_memory_layout.c`); the 16 B-aligned pool header stays 6208.

**Unused code removed** (internal headers, test hooks and the example only; the public headers lose nothing but `aec_debug.h`):
- `WAV_IO_WRITER_STYLE` default in `example/wav_io.h` (the shared header already defaults to the AEC style).
- `aec_config_defaults()` (internal one-line alias of `aec_config_from_level(..., AEC_LEVEL_BALANCED, ...)`); tests call `aec_config_from_level()`.
- Test hooks `aec_testing_get_mem_breakdown()` and `aec_testing_init_ex()` (21 `aec_testing_*` hooks remain); the one test that used them calls `aec_get_mem_breakdown()` and `aec_init_ex()`.
- `aec_render_stats()`, `AecRenderStats`, `AecResInput.render_sum_sq` and `.render_peak_sq`, `aec_res_split_floor()`. One behaviour change in the stand-alone gain stage: `aec_res_process()` with a NULL `render_hop` now returns NULL (it used the removed statistics fields). Every in-tree caller sets `render_hop`; the stage is not in the pool or the layout hash.
- `aec_state_transparent_mode_active()` (always 0; `ree_estimate()` gets the literal `0`), the unused `echo_psd` parameter of `aec_state_update()`, the unused `render_sq_scratch` parameter of the consistency detector in `filter_analyzer.c`.
- Return values nobody read: `aec_post_run()`, `delay_aec_accumulate_ex()`, `filter_delay_update()`, `epc_force_delay()` and `shadow_copy_update()` are `void`; `ShadowCopyDecision` (including the never-read `reverse_copy`) and `RenderActivityResult.is_stationary` (read from `RenderActivity` instead) are gone.
- Tests: `test/baseline3_mono.h` is folded into `test/test_split_entry3.c`, `test/profile_timer_count.h` and `profile_timer_stub.h` are one `test/profile_timer_fixtures.h`, `test/modules/parity_detectors.c` is `test/parity_detectors.c`.
- `docs/c_python_function_map.md`: the rows of the removed functions are gone, `aec_config_from_level` carries the removed alias's gap labels, `AecState.transparent_mode_active` is listed as Python-only, and the statistics block is recounted.

## [Unreleased] — 2026-11-14 — Review round: shared pool and hash helpers, one level-apply helper, strict `aec_wav` integers, fail-closed pin gate

**Output and layout unchanged.** `aec_get_mem_size()`, `aec_get_mem_requirements()` (including `build_flags_hash`, `c8cb8b58` for the default config) and `aec_get_mem_breakdown()` are identical before and after on 1,792 configurations (4 grids x mono/array x 4 residual modes x 3 delay modes x `delay_num_filters` x 4 levels x CNG) at `SIMD=1` and `SIMD=0`; the five `AEC_TEST_BASELINE` streams (`baseline_round2`) compare with 0 mismatches, `META` pool lines included.

- `aec_array.c` carves its pool with audiolib's `AcPoolCursor` / `ac_pool_carve()` (its private `PoolCursor` / `pool_carve()` was a copy) and sizes the flow buffers with `ck_field_size_reps()`; `aec_build_flags_hash()` folds its carve text through `ac_fnv1a_str()`. A lane that fails to build no longer calls `aec_destroy()` on the lanes before it: every lane lives in the caller's pool, so there is nothing to release.
- `aec_sg_apply_level()` (new, internal) holds the three level lookups, the three suppressor setters and the ramp; `aec_set_level()` and `aec_res_set_level()` call it instead of each carrying a copy. The duplicated pre-checks in `aec_set_level()` are gone (the helper validates before it writes).
- `aec_config_from_level(NULL, ...)` returns instead of crashing; the `AecStatus` comment says what a refused call does (instance untouched; view getters zero their `out`).
- `aec_wav`: `--fft-size`, `--delay-num-filters` and `--fixed-delay` parse as whole decimal integers (`aec_cli_parse_int()` in `example/cli_level.h`); a typo exits 2 with "is not an integer" instead of reading as 0. `test/cli_delay_flags.sh` checks six malformed values.
- `test/pyc_parity/run_parity_e2e.sh` stops with a setup error (exit 2) when the pin file lists no inputs; an emptied pin file used to pass the input check without verifying anything.
- Unused includes removed (`aec_internal.h` from `aec_array.c`, `aec_balanced_config.h` and `utility/fast_math.h` from `aec_res.c`; object code identical).
- `docs/c_python_function_map.md`: `pool_carve` row removed, `aec_sg_apply_level` row added.

## [Unreleased] — 2026-11-14 — Docs only: which spectrum the denoiser reads

Documentation only; no source, header or output changes. `docs/reference/integration.md`, `docs/html/reference/aec_usage.html` and `docs/sdk_migration_output_guard.md` now say the same as `aec.h`: the shipped mono chain hands `base_spec` to the denoiser (it estimates on the spectrum its gain is multiplied on), the shipped 4ch chain hands `error_spec`, the two differ only on output-guard hops, and `error_spec` / `formed_hop` stay the analysis seam. The usage example shows `base_spec`.

## [Unreleased] — 2026-11-14 — Finish round: default-tuning pin, header comments, figures corrected

**Library output and pool unchanged** (the five baseline streams written from this tree compare with 0 mismatches at `SIMD=1` and `SIMD=0`; against the streams of f6e846f only the 21 `META` pool lines of `split_entry` differ).

- `test_config_validation.c` pins the default tuning: FNV-1a-32 over the bytes of the `AecTuning` that `aec_testing_tuning_defaults()` builds, for 8 / 16 / 48 kHz at the four levels (12 hashes; 504 checks, was 492). A one-unit change of a default or of a level's floors fails the rows it touches without the external baseline directory. Mutations tried: `warmup_frames` 100 -> 101 (12 rows fail), aggressive far-active floor -38 -> -37 dB (3 rows fail). Re-pin a row only together with a deliberate default change, and re-capture the baseline streams with it.
- Eval harness (no library change): the 90-case A/B runners and `python/eval_manifest90.py` resolve a stem to its scenario directory, else to `_quarantine_non_official/` (the same 90 clips as before), and verify `eval/manifest_90case.sha256` before rendering; `SELF_CHECK=1` allows a null run; `build_manifest_90case.py` no longer claims to reproduce the checked-in set and requires `--out`/`--stems-out`.
- `aec.h`: three comments no longer name internals (`AecState`, `a->far_hop`, `PBFDKF`); the preprocessed header is unchanged.
- Entry corrections below: AecTuning has 42 fields after the dead-state entry (it said 45 -> 43); the FULL array pool of the dead-state entry is -22,224 / -38,608 / -71,376 B (STATE and ESTIMATE -22,208 / -38,592 / -71,360); the `aec_testing_*` count is 23 at the head of 2026-11-14.

## [Unreleased] — 2026-11-14 — `AEC_LAYOUT_VERSION` 2; pool, `sizeof` and `libaec.a` figures re-measured

**API change (descriptor only); output unchanged.** The baseline streams (`AEC_TEST_BASELINE`, written by the tree before this change) compare with 0 mismatches at `SIMD=1` and `SIMD=0` over all five files; the only lines that differ are the 21 `META` pool-size lines of `split_entry` (`mem_size_*` x5, `aec_res_mem_size`, `sizeof_Aec`, x3 grids).

- `AEC_LAYOUT_VERSION` 1 -> 2, once for the whole round (the carved layout of the single channel and of the array changed in the entries below). The `build_flags_hash` carve text moves with it: `cfg-v1` -> `cfg-v2`, `swin` (the synthesis window, carved before `ola` in the single channel and first in the array's flow buffers) and `tables` (the decoded ENR/SU tables in the single channel's suppression carve and in the array's `AecRes` block); a lane carries neither. FNV-1a-32 of the text 3,614,935,402 -> 3,368,782,680 (LP64; `ac_memreq_abi_hash` is the identity there). A descriptor queried from an earlier build is refused by `aec_init_ex()`; `aec_init()` and `aec_create()` never compare one.
- `test_memory_layout.c` pins `AEC_LAYOUT_VERSION` (and, on LP64, the hash) as literals, next to the existing `sizeof(Aec)` and pool pins.
- Pool bytes (balanced, `MATCHED` n = 5, 64-bit host; mono is the same for every `residual`, `SIMD` and `AEC_NO_ARRAY`):

  | grid | mono before -> after | array STATE | array ESTIMATE | array FULL |
  |---|---|---|---|---|
  | 16k/256 | 343,760 -> 239,920 | 616,912 -> 595,744 | 712,448 -> 691,280 | 720,352 -> 701,280 |
  | 16k/512 | 456,640 -> 352,288 | 920,464 -> 883,936 | 1,110,208 -> 1,073,680 | 1,124,256 -> 1,091,872 |
  | 48k/1024 | 1,078,560 -> 776,224 | 2,062,960 -> 1,995,712 | 2,441,120 -> 2,373,872 | 2,470,528 -> 2,411,520 |
  | 8k/256 | 249,856 -> 211,552 | n/a | n/a | n/a |

  `sizeof(AecConfig)` 228 -> 48 B; the pool's `Aec` header 6,368 -> 6,208 B. Per-filter step of `delay_num_filters`: single channel 5,728 -> 11,264 B (16 / 8 kHz), 23,552 B (48 kHz) because the reference ring now grows with the bank; the array's shared estimator 5,120 B. Single-channel pool without the shadow filter (internal switch) 212,032 B at 16k/256.
- `libaec.a` `__TEXT` (`size` summed over the members; Apple clang, arm64; the target toolchain is not measured): `SIMD=1` 184,758 -> 164,227 B, `SIMD=0` 166,874 -> 145,667 B; `AEC_NO_ARRAY=1` 163,426 -> 142,015 B and 148,714 -> 126,643 B.
- Docs: every pool, lane, fusion, `AecRes`, per-mode and per-`delay_num_filters` figure of `c_impl/STATIC_MEMORY.md`, `c_impl/README.md`, `docs/reference/{lifecycle,configuration,wav_cli}.md`, `docs/delay_estimator_design_zh_TW.md` and the reference HTML pages replaced by fresh measurements (`aec_get_mem_breakdown()` of the shipped library; `AEC_TESTING` hooks for the shadow and observer switches); `STATIC_MEMORY.md` describes the two tables built into the pool; `delay_buffer_ms` no longer named as a read field.
- **Supersedes** the pool, lane, `AecRes`, `libaec.a` and `sizeof` figures of every earlier entry below.

## [Unreleased] — 2026-11-14 — Dead state removed, the single-channel reference ring sized from the estimator reach, `delay_buffer_ms` deleted

**Output unchanged** (every hop hash of the baseline streams is identical at `SIMD=1` and `SIMD=0`, after every item; only the `META` pool lines differ). No public symbol changed.

- Deleted state nothing read: the pre-echo lag aggregator and the matched filters' accumulated / instantaneous error buffers (`DaPreEcho`, `da_pre_echo_*`, `DA_ACC_ERR_SIZE`, ...); the PBFDKF `Q`, `Q_high`, `Q_low`, `R` arrays with `filter_q_high` and `ShadowCopyDecision.boost_q` (`kalman_q_high`, `kalman_q_low` and `delay_buffer_ms` leave `AecTuning`: 42 fields remain); five unread `AecPostRunIn` fields; nine store-only fields (`LinearFilterSelect.block_win`, `max_erle_lf_log2`, `EpcEvent.source`, `FilterDelayEstimate.quality`, `last_lf_band` / `first_hf_band`, ...); `epc_mu_floor`; the shadow filter's `sqrt_hann` (a follower does not carve one). `AECB_SG_LAST_LF_BAND` / `AECB_SG_FIRST_HF_BAND` leave `aec_balanced_config.h`.
- `delay_buffer_ms` is deleted. The single-channel `MATCHED` reference ring is `delay_aec_max_delay_samples(rate, n) + hop` samples (`4 * ((n-1) * 384 + 501)`, x3 at 48 kHz; 8,148 at n = 5 and 16 kHz), the largest delay the estimator can report, instead of a 2048 ms ring. Evidence: the new ring against the old one is bit-identical on 683 cases (synthetic, `max_delay_ms`, `delay_num_filters` 1..5, `FIXED`, 154 real recordings, 8 kHz, stress) and the reach ring on 1,138 boundary cases; the same ring one sample short differs on 180 of them. `test_delay_num_filters` pins the reach per grid and bank size and checks an echo at the bank edge is applied. The array ring is unchanged (`max_delay_ms + 2 hops + 1`).
- Pool (single channel; the figures before this entry are those of the previous one): 346,896 -> 239,920 B at 16k/256 (pre-echo state -4,768, Kalman arrays -2,144, store-only fields -1,072, shadow `sqrt_hann` -1,024, reference ring -97,968), 462,848 -> 352,288 B at 16k/512, 1,090,912 -> 776,224 B at 48k/1024, 252,992 -> 211,552 B at 8k/256; array -22,208 / -38,592 / -71,360 B (STATE and ESTIMATE) and -22,224 / -38,608 / -71,376 B (FULL: the `AecRes` block is also 16 B smaller; the figures follow from the final pools in the "Pool bytes" table of the `AEC_LAYOUT_VERSION` entry above, minus the window and table entries) at 16k/256, 16k/512, 48k/1024. `sizeof(Aec)` 6,368 -> 6,208 B.
- `python/aec.py` is unchanged: the Python twin still carries these fields and sizes its ring `max(delay_buffer_ms, max_delay_ms + 4096)`. `docs/c_python_function_map.md` marks the affected rows `partial` (gap D-16); outputs agree at the Python defaults, where the ring is not shorter than the reach.

## [Unreleased] — 2026-11-14 — The synthesis window and the ENR / SU threshold tables are built at init

**Output unchanged** (hop hashes identical at `SIMD=1` and `SIMD=0`; the `META` pool lines differ). Public API unchanged apart from where `aec_synthesis_window()` points.

- The synthesis window is a 513-float half table plus a mirror / stride built by `aec_init()` / `aec_create()` into the pool (no libm): one `fft_size`-float window per single-channel instance and one per array, none in a lane. `aec_synthesis_window()` returns that pointer, so a borrower (for example an `AcIstft` configured with `.window = w`) must not outlive the `Aec`. Half-table values are double-precision `sqrt(0.5 * (1 - cos(2 pi i / n)))` rounded to float, bit-identical to the three tables they replace.
- The ENR / SU gain-threshold tables are run-length coded (`AECB_*_SG_RUN_END`, `_RUN_VAL`, `_RUNS`) and decoded into the pool by the single channel (suppression carve) and by the array's `AecRes` block; the decoded copies survive `aec_res_reset()` and `aec_res_set_level()`. The EMR table stays a flash constant. The 8 kHz row aliases the 16k/512 code (decode clipped to 129 bins).
- `aec_balanced_config.h` loses `AECB_*SYNTH_WINDOW` and the `AECB_*_SG_{NEAREND,NORMAL}_ENR_{TR,SU}` arrays (108,105 -> 44,494 B of source) and gains `AECB_SYNTH_HALF_N`, `AECB_SYNTH_WINDOW_HALF` and the run-length code. `gen_aec_balanced_config_h.py` asserts `decode(encode(x)) == x` bit for bit, that every captured window is rebuilt from the half table, and writes the test-only `c_impl/test/aec_ref_tables.h` (full windows and tables per grid, never linked into `libaec.a`), which `test_synthesis_window`, `test_rate_structural` and `test_config_validation` use as an independent oracle; `make test-header-fresh` compares both files.
- Pool: +1,024 / +2,048 / +4,096 B (window) and +2,112 / +4,160 / +8,256 B (tables) per mono instance at 16k/256, 16k/512, 48k/1024; array +1,040 / +2,064 / +4,112 B (STATE, ESTIMATE) and +3,152 / +6,224 / +12,368 B (FULL). `libaec.a` flash: `__const` -19,032 B, `__text` +1,412 B (host, `SIMD=1`).

## [Unreleased] — 2026-11-14 — `AecConfig` has 12 fields; the tuning is internal (`AecTuning`)

**API change; output unchanged** (all five baseline streams identical at `SIMD=1` and `SIMD=0`, `META` pool lines included).

- `AecConfig` is `sample_rate`, `fft_size`, `n_mics`, `level`, `residual`, `delay_mode`, `fixed_delay_samples`, `delay_num_filters`, `enable_cng`, `capture_proxy_channel`, `delay_backward_quarantine_enabled`, `delay_backward_quarantine_s` (228 -> 48 B). `level` is new and replaces the three `min_gain_floor_*` fields. `aec_config_from_level()` zero-fills the struct and stores the level; a zero-filled `AecConfig` is `AEC_LEVEL_MILD`; a level outside 0..3 is refused by every entry that takes a config. `aec_set_level()` also stores the level.
- The other 46 fields (`filter_length`, `mu`, `delta`, `enable_shadow`, `enable_highpass`, `warmup_frames`, `epc_*`, `max_delay_ms`, `delay_est_*`, `highpass_cutoff_hz`, `saturation_*`, the `live_observer_*` set, `cng_n2_windowed_psd`, ...) live in the library-internal `AecTuning` (`src/aec_tuning.h`), built by `aec_resolve()` from `(sample_rate, fft_size, n_mics, level)` and validated there; the filter length follows the rate (416 / 832 / 3072 samples at 8k / 16k / 48k). `aec_config_defaults()` is internal; callers use `aec_config_from_level()`. `epc_mu_floor` is deleted.
- `aec_wav` drops `--no-shadow`, `--no-hpf` and `--delay-est-period`; `bench_rtf --live-observer` needs `make bench AEC_TESTING=1`.
- Tests reach the internal tuning through `AEC_TESTING` hooks (`aec_testing_*`, `test/aec_test_tuning.h`); the Makefile variable `AEC_HOOK_GOALS` turns `AEC_TESTING` on for the goals that need them. `test-no-testing-symbols`: the testing archive exports every hook, the production archive none.

## [Unreleased] — 2026-11-14 — `Aec` is opaque; `Aec* aec_create(const AecConfig*)`; four public headers

**API change; output unchanged** (baseline streams identical at `SIMD=1` and `SIMD=0`, `META` lines included).

- `aec.h` declares `typedef struct Aec Aec;` only; the body is `src/aec_impl.h`. `Aec x;`, `sizeof(Aec)`, `calloc(1, sizeof(Aec))` and field access outside the library no longer compile.
- `Aec* aec_create(const AecConfig*)` makes one 16-byte-aligned allocation of `aec_get_mem_size(cfg)` bytes and builds the instance in it (NULL for a NULL or invalid config or a failed allocation). `aec_destroy()` frees that block: the pointer is dead afterwards, and the caller no longer calls `free()` on it. `aec_init()` pool instances are unchanged (`aec_destroy()` clears `ready`; a destroyed handle is refused).
- `include/` keeps four headers: `aec.h`, `aec_internal.h`, `aec_balanced_config.h`, `aec_debug.h`. The other 25 moved to `src/`. `aec_debug_set_level()` / `aec_debug_set_log()` (stored, never read) and the `aec_wav` flags `--debug-level` / `--debug-log` are deleted.
- `DA_NUM_FILTERS` -> `AEC_DELAY_NUM_FILTERS_MAX` (no alias); `AEC_DELAY_FILTER_SIZE`, `AEC_DELAY_FILTER_INTRA_SHIFT`, `AEC_DELAY_DOWN_SAMPLING_FACTOR`, `AEC_CNG_SEED`, `AEC_PSD_SCALE_F`, `AEC_PSD_SCALE_INV_F`, `aec_level_name()` and `AecStageTiming` are public in `aec.h`. `AecDebugStatus` gains `n_partitions` and `delay_period_hops` (read-only; the AIAEC deployed-lane gate reads them).
- Tests: about 144 call sites in 35 files moved to `Aec* x = aec_create(&cfg)`; `test_lifecycle`, `test_handle_ready`, `test_heap_alignment` and `test_aec_array` were rewritten where the meaning changed (110 array create / process / reset / destroy cycles; one block, 16-byte aligned, one release). `make test-header-cpp` also compiles `aec.h`, `aec_internal.h` and `aec_debug.h` each alone.

## [Unreleased] — 2026-11-14 — Small tidy: shared helpers, dead macros, build and test wiring, stale docs

**Output unchanged** (the five baseline streams compare with 0 mismatches at `SIMD=1` and `SIMD=0`; no public symbol changed).

- `aec_duty_configure()` is the one seconds-to-hops conversion of the duty gate (`aec.c` and `aec_array.c`); the aec_init and `aec_reset()` scalar-state lists are one function (`aec_reset_scalar_state`; `test-reset-parity` and the baseline streams pass); one grid lookup serves `aec_resolve_signal_grid`, `aec_is_valid_sample_rate` and `aec_default_fft_size`.
- `aec_post_compute_*`, `aec_post_apply_output`, `pbfdaf_get_time_domain_filter` and `pbfdaf_scale_filter` are `static`; the unread `AEC_POST_DELAY_ADJ_BUFFER_FLUSH` / `_NEW_DETECTED`, `AECB_ENABLE_CNG`, `AECB_SG_SPLIT_FLOOR_FAR_ACTIVE` and `AecBalancedRateDims.ree_hop_size` are gone (generator edited, `aec_balanced_config.h` regenerated).
- Makefile: one goal-list variable (`make -pn` rule sets identical for `SIMD=0/1` x `AEC_NO_ARRAY=0/1`), no `dirs` target, `make debug` adds only `-g`, `$(PYTHON)` in the detector-parity recipes, `test-lifecycle` and `test-docs-smoke` (both in the SIMD-independent tier), help lists the nine missing targets and the `--live-observer` / `--wav` options of `bench_rtf`. `test_wav_io_shim.c` and `test/modules/parity_saturation.c` are deleted.
- `eval/run_py_ab.sh` renders with `--filter 832` (52 built a one-partition filter); `eval/manifest_90case_stems.txt` still lists 29 stems that are missing from their scenario directory.
- Docs: stale counts, `aec_process_core` and missing-document references repointed, `--debug-level` / `--debug-log` documented as stored-only, `libaec.a` text figures re-measured with their command.

## [Unreleased] — 2026-11-13 — Finish round: duty-gate / ring structural test, stale numbers and wording fixed

**Output unchanged** (no change under `src/` or `include/` apart from one comment in `aec_post.h`).

- `test_duty_census.c` gains section 8: `aec_duty_end/begin` stepped with `hold_hops = 3`, `period_hops = 4` (arms on the hold-th stable estimate, then runs one hop in four, a moved estimate disarms) and an `aec_ring_write/aec_ring_read` round trip that wraps at a nonzero offset. Three mutants that only the retired baseline files caught (duty hold `>=` -> `>`, period wrap `>=` -> `>`, `aec_ring_read` tail copied to `dst` instead of `dst + first`) now fail this test.
- Docs: `sizeof(AecConfig)` 232 -> 228 B; `libaec.a` `__TEXT` 184,765 B (`AEC_NO_ARRAY=1`: 163,433 B) at `SIMD=1`, 166,881 B (148,721 B) at `SIMD=0`; broken `§6.2` anchor in `integration.md`; the `sdk_migration_output_guard.md` example no longer passes a removed argument; `integration.md` §8.9 gives one example call of `aec_apply_external_realign()`; the HTML reference pages show the 2-argument `aec_set_level`, no longer present `aec_level_*` / `aec_cng_fill` as public, carry symbol names instead of drifting line numbers, and count 57 `AecConfig` fields.
- `parity_aec_e2e.c` and the Makefile help text document numeric levels only; the `test-header-cpp` message no longer mentions legacy names.

## [Unreleased] — 2026-11-13 — Public header trims, legacy aliases and `enable_delay_est` removed, duty-gate / ring helpers merged, single-channel dead allocations removed

**Output unchanged.** The five entry-point baseline streams (split_entry 1-3, aec_array 1-2) compared against the files the pre-split API wrote (AEC 2c8ced3 `git archive`, SIMD=1 and SIMD=0): every line identical except the pool `META` lines (`mem_size_*`, `sizeof_Aec`) of split_entry, which this entry moves on purpose. `make test-full` (SIMD=1), `make test` (SIMD=0) and `make AEC_NO_ARRAY=1 test` (SIMD=1, 0) pass.

- **Header.** `aec_process_linear_shared_far_from`, `aec_level_floor_db` / `aec_level_dt_floor_db` / `aec_level_split_floor_enabled` and `aec_cng_fill` moved from `aec.h` to the new library-internal `include/aec_internal.h` (`aec_apply_cng_spec` stays public). Deleted: `aec_get_residual_gain` (read `aec_get_residual().res_gain`), `aec_array_duty_hops_total` / `_run` (read `AecDebugStatus`), `aec_array_pending_delay_candidate` (now the test hook `aec_testing_array_pending_delay_candidate`; `test-aec-array` and `test-aec-array-2` re-invoke make with `AEC_TESTING=1` like `test-aec-array-flow`), the `aec_testing_array_*` clause of the `AEC_NO_ARRAY` preamble, `AEC_PRESET_*` / `AecPreset`, `Aec::scr_e2coa_pre` / `scr_erl_arr`.
- **`enable_delay_est` and `aec_config_resolve_delay()` removed.** `delay_mode` is the only selector; `aec_wav --no-delay-est` now sets `delay_mode` (FIXED when `fixed_delay_samples >= 0`, else EXTERNAL_ALIGNED). Callers that used `enable_delay_est = 0` as the only selector now name the mode.
- **Duty gate and reference ring shared.** `Aec::duty_*` and the array flow's copy are one `AecDutyGate` with `aec_duty_reset/begin/end` and one `aec_duty_leak_db()`; the single-channel ring and the array ring use `aec_ring_write/read`. Mutating `aec_duty_begin` (inverted `run_filter`) changes 4 of the 5 baseline streams, mutating the ring read changes all 5.
- **Single-channel dead allocations.** `x2_past` and the `X_buf` fallback of the render-reverb term, `rh_st` (the echo-generating window is on in the balanced configuration), stationarity-history rows beyond the ring's length, the shadow filter's own `X_buf` / `x2_cache` / `near_spec` / `error_spec_windowed` / `spec_scratch` (the shadow is now a follower of the main filter in every instance; shadow vs main far history was compared every hop over about 1.5 million hops of the test suite with zero differences), and, for `STATE` array lanes, the residual-echo estimator and comfort-noise arrays (`ree_init()` and the reverb models take NULL storage). `AEC_LAYOUT_VERSION` stays 1 (never released); `build_flags_hash` tokens updated.
- **Pool sizes** (balanced, MATCHED n=5, 64-bit host): mono 16k/256 366,048 -> 343,760 B, 16k/512 494,768 -> 456,640, 48k/1024 1,162,784 -> 1,078,560; array 16k/256 STATE/ESTIMATE/FULL 710,576/716,832/724,736 -> 616,912/712,448/720,352, 16k/512 1,106,288/1,118,688/1,132,736 -> 920,464/1,110,208/1,124,256, 48k/1024 2,433,104/2,457,792/2,487,200 -> 2,062,960/2,441,120/2,470,528; `sizeof(Aec)` 6,400 -> 6,368. `libaec.a` text (SIMD=1) 186,105 -> 184,829 B, `AEC_NO_ARRAY=1` 164,509 -> 163,429 B.
- **Supersedes** the pool figures and the `sizeof(Aec)` statements of the earlier 2026-11-13 entries below ("Review fixes: residual getters need a completed hop...", "Array lanes 1-3 reuse lane 0's far-only analysis", "Array lane pools shrink by 22 %", "A handle is usable only after a successful create / init", "Review fixes: array beam validation, destroyed handles...") and of the 2026-11-12 array entry: they describe the state before this entry (`sizeof(Aec)` is 6,368 B and `sizeof(AecConfig)` 228 B at this head). Every mention of `aec_get_residual_gain()` in the 2026-11-12 and 2026-11-13 entries below names the getter deleted here. Baseline note: the mono streams come from the release build, the array streams (`test-aec-array`, `test-aec-array-2`) from an `AEC_TESTING=1` build; the lines are identical either way.
- Tests: `test/res_gain_of.h` (view-based replacement of the deleted getter); `check_handle_entries.py` also reads `aec_internal.h`; the delay-mode translation tests are replaced by the delay-mode defaults and the rejection matrix; `test_spectral_reuse` skips the spectra a follower shadow does not have; `test_memory_layout` pins updated. Python `test_preset_arg.py` reads `AEC_LEVEL_*`.

## [Unreleased] — 2026-11-13 — `aec_set_level(Aec*, AecLevel)`: the ramp is an internal 100 ms; `--level` takes numbers; the array's gain stage can be replayed through an AEC_TESTING hook

**API change; output changes only where a level is changed at run time.** Every instance built at a level, and every run that never calls `aec_set_level()`, is unchanged.

- `aec_set_level(a, level)` replaces `aec_set_level(a, level, ramp_ms)`. The far-active floor always walks to its new value linearly in dB over `AEC_LEVEL_RAMP_MS` (100 ms, `src/aec_res_internal.h`); the DT floor and the split-floor flag still step at once. A caller that passed another time (0 for "next hop", up to 60 s) no longer can. `aec_res_set_level(r, level)` (internal) follows the same rule. `suppression_gain_set_split_floor_far_active_db(sg, db, ramp_ms)` keeps its `ramp_ms` (`test_runtime_preset.c` drives it with 0 ms and with the half-hop inputs).
- Level arguments are numbers. `aec_wav`, `bench_rtf`, `parity_aec_e2e` and `pyc_parity/aec_run` read `--level` through `example/cli_level.h` (`aec_cli_parse_level()`): a whole decimal string 0..3 (0 mild, 1 balanced, 2 aggressive, 3 no_floor); a name, an empty or padded string, a suffix (`1x`) or a number out of range prints an error and exits 2 (it was a name table, `bench_rtf` and `aec_wav` also took names). `pyc_parity.py` and `run_parity_e2e.sh` translate their level names to the number for the C side only; the Python twins keep names. `README.md`, `docs/maintenance/development.md`, `docs/reference/wav_cli.md`, `docs/html/index.html` and `eval/run_c_ab.sh` use `--level 1`.
- `aec_test_hooks.h` (AEC_TESTING build only): `aec_testing_res_mem_size()`, `aec_testing_res_init()` and `aec_testing_res_process()` run the array's gain stage (`src/aec_res.c`) on its own over an `AecResidualView`, fed the way the array feeds its own stage. `test_aec_array_flow.c` (now 713 checks) shows the standalone stage equals the array's own `res_gain` on every one of 400 hops at 16k/256, 16k/512 and 48k/1024, and that a stage at another level differs on 156-232 hops (the comparison can fail). `make test-no-testing-symbols` passes in both directions (14 `aec_testing_*` symbols in the AEC_TESTING archive at the time of this entry, 23 at the head of 2026-11-14; none in the production one). Audio_ALG's `pyc_parity_pipelines.sh` links that archive into one extra driver (`pyc_drv_4ch_replay`) so the nine `l2.res4.*.replay` rows recompute G_res again.
- Tests: `test_runtime_preset.c` drives the 0 ms behaviour through the suppressor and checks a completed `aec_set_level()` ramp lands on the constructed float; `test_aec_res.c` checks the walk and its landing; `test_aec_array.c`, `test_aec_array2.c`, `test_handle_ready.c`, `test_zero_heap_aec.c`, `test_header_cpp.cpp` and the `baseline2_*.h` harness follow the new signature (a level event carries only the level).
- Baseline streams (`AEC_TEST_BASELINE`, files captured from the previous version at `SIMD=1` and `SIMD=0`): `aec_array`, `split_entry` and `split_entry3` are byte-identical (0 mismatches of 138,315 / 76,527 / 5,910 lines); `aec_array2` and `split_entry2` differ only in the cases that contain a level event (`levels`, `level0`, `level1`, `level3`, `fixed`), only in the `gain`, `apply` and `out` lines and only from the first event on (266 of 6,903 and 1,110 of 22,465 lines); every other case matches. Those cases are excluded from the comparison for this change; the next baseline is captured from this version.

## [Unreleased] — 2026-11-13 — A handle is usable only after a successful create / init (`Aec.ready`); baseline capture fails loudly

**Output unchanged** (all five baseline streams are byte-identical to the ones captured before this entry at `SIMD=1` and `SIMD=0`; `sizeof(Aec)` and every pool size unchanged).

- `Aec.destroyed` became `Aec.ready` (same byte): set as the last statement of a successful `aec_create()` / `aec_init()` (single-channel, array, lane), cleared by `aec_destroy()`. A NULL, zero-filled or never-built handle, the handle a failed `aec_create()` left behind and a destroyed handle are refused with the same codes. `aec_unusable()` moved to `src/aec_array_internal.h`; `aec_array.c` no longer repeats the raw test.
- `make test-handle-ready` (new, `test/test_handle_ready.c`): all 29 `aec.h` entry points that take an `Aec` handle on a zero `Aec` (static, stack, calloc, a zero pool cast to `Aec*`), after a failed `aec_create()` (also over a destroyed handle) and after `aec_destroy()` for single-channel and array, heap and caller pool; a live handle of each kind is the control. `test/check_handle_entries.py` compares the list of calls with the prototypes in `aec.h`. Each of three mutations (the predicate without `ready`, a missing `ready = 1`, a missing `ready = 0`) was seen to fail it.
- Config validation: `aec_validate_config()` and `aec_resolved_config()` are one `aec_resolve_valid_config()` (the delay is resolved once); the quarantine range checks are shared by the `AEC_NO_ARRAY` and the array build; `aec_array_impl_config_ok()` is declared only where it exists.
- `test/baseline_stream.h` replaces five copies of the baseline loader. Asking for a baseline (`AEC_TEST_BASELINE`, `AEC_TEST_BASELINE_WRITE`) and not getting it (missing or unreadable file, unwritable directory, host build without a `simd0` / `simd1` name, `G2_ONLY` set) is a failure, not a pass with a note. `test_split_entry2.c` always requires its apply cross-check to have run, `test_aec_array2.c` requires its view invariants to have run on at least one hop.
- `test_memory_layout.c`: the 216-configuration array walk fails on a refused size and requires all 216 to be checked on every ABI; the array pins print a skip line on 32-bit.
- `gen_aec_balanced_config_h.py`: the multi-rate comment no longer claims a single including TU (comment-only change of `aec_balanced_config.h`); the EMR alias list is built once.

## [Unreleased] — 2026-11-13 — One-off pre-split baselines retired; the entry-point tests keep an optional line-stream baseline

**No library change.** The files that the pre-split API (AEC 2c8ced3, audiolib 43d562a, Audio_ALG cf023eb) wrote for the entry-point split
and the array fold were one-off evidence. Before they were deleted the same recipe was re-run on pristine `git archive` copies of those
commits (`SIMD=0` and `SIMD=1`, Apple arm64 / Apple clang 17 / NE10) and every file was reproduced byte for byte; the tests of the new API had
consumed all of their lines with 0 mismatches: single-channel set 1 (76,527 lines), set 2 (22,465) and set 3, the bad-relock judgments that
free-running scenes produce (5,910), and array set 1 (138,315) and set 2 (6,903). The files, their writers, `build_dump.sh`, the READMEs and
`PORTING.md` (the map from the old `test_mcab.c` groups to the array tests) are no longer in the tree.

- `test/golden/` is removed. The source pin of the end-to-end parity gate moved to `test/parity_e2e_inputs.sha256` (`run_parity_e2e.sh`, `make test-parity-e2e`).
- `test_split_entry.c`, `test_split_entry2.c`, `test_split_entry3.c`, `test_aec_array.c` and `test_aec_array2.c` keep every structural check and
  no longer look for a file. Their hashed line stream can be captured and compared: `AEC_TEST_BASELINE_WRITE=<dir>` writes
  `<dir>/<test>_simd{0,1}.txt` (`simd0` = `SIMD=0`, `simd1` = NEON build, nothing on other hosts), `AEC_TEST_BASELINE=<dir>` compares every line.
  The next change captures its baseline from this version first. The pool-size lines (`mem_size_*`, `sizeof_Aec`, `aec_res_mem_size`),
  skipped while comparing with the pre-split files, are now compared. The scene-expectation checks of `test_split_entry3.c` (and the apply
  cross-check count of `test_split_entry2.c`) run on the NEON and scalar builds without a file, as before.
- `make test-split-entry-3` is new (the third set; mono only, also under `AEC_NO_ARRAY=1`).

## [Unreleased] — 2026-11-13 — Follow-up: non-finite input and destroyed handles are documented and tested as the code behaves

**Output unchanged; no code path changed** (comments and tests only). A non-finite input sample makes `aec_process_linear()` answer `AEC_ERR_ARG` before anything is written: the previous hop's views stay readable, its pending linear frame stays pending and the
stage-timing record is untouched. The docs (`aec.h`, the C API / sequence pages, `lifecycle.md`, `integration.md`, `configuration.md` and the Audio_ALG `mcab` references) said the previous views were invalidated, the valid flag cleared or the timing zeroed. A `beamformed_error` with
a non-finite bin is `AEC_ERR_ARG` (not `AEC_ERR_DSP`) and resets nothing in the array.

- `test_aec_array.c` `check_destroyed_refuses()` (array and single-channel, heap and static) now also checks `aec_process_linear_shared_far_from()` -> `AEC_ERR_ARG` and that `aec_get_residual()`, `aec_get_linear()` and `aec_get_array()` leave a zeroed record with
  only `struct_size` set; `test_lifecycle.c` and `test_memory_layout.c` repeat the single-channel part (the latter also runs under `AEC_NO_ARRAY=1`). Each assertion was checked to fail with its fix removed.
- Comments of the `aec_carve()` failure paths no longer claim an FFT handle is released (every handle lives in the carved memory; the heap path frees the arena).

## [Unreleased] — 2026-11-13 — Generated balanced-config header: tables that are prefixes of another table are emitted once

**Output unchanged.** `python/diag/gen_aec_balanced_config_h.py` emits a table as a `#define` naming an earlier table when the table is bytewise the leading floats of it (a consumer reads only `table_len` entries). Per-bin tables are
exactly such prefixes: every `*_EMR_TR` table of every grid is the constant 0.3 (one `AECB_SG_EMR_TR_COMMON[513]` serves all eight), the 8 kHz `ENR_TR` / `ENR_SU` tables start with the 16 kHz ones, and the 8 kHz and 16 kHz/256
synthesis windows are equal (`AECB_SYNTH_WINDOW_256`, emitted under `AECB_ENABLE_8K || AECB_ENABLE_16K_256`). The per-grid `AECB_ENABLE_*` defaults moved above the first table that depends on them; no macro was renamed. A table that is not a
prefix is still emitted in full, and `make test-header-fresh` regenerates the header and diffs it against the checked-in copy.

- `include/aec_balanced_config.h`: 140,411 -> 108,764 B (540 -> 550 lines).
- `libaec.a` `__TEXT` (host, sum of the members' `__TEXT` sections): 196,661 -> 187,405 B at `SIMD=1`, 178,481 -> 169,225 B at `SIMD=0`; `__TEXT,__const` 33,284 -> 24,028 B in both (`__text` unchanged).
  The pool sizes do not move. Entries below that quote a `libaec.a` size were measured before this one, except the `AEC_NO_ARRAY` entry, which quotes the sizes after it.
- Tests: the four array / split-entry goldens (`golden*_simd{0,1}.txt`) are byte-identical and still match at `SIMD=1` and `SIMD=0`.

## [Unreleased] — 2026-11-13 — `AEC_NO_ARRAY`: a single-channel library without the four-microphone array instance

**Output unchanged.** The default build (`AEC_NO_ARRAY=0`) is the same code as before bit for bit: `test-split-entry`, `test-split-entry-2`, `test-aec-array`, `test-aec-array-2` match their goldens at `SIMD=1` and `SIMD=0`, and with `AEC_NO_ARRAY=1`
the two single-channel goldens still match bit for bit.

- **`make AEC_NO_ARRAY=1` (`-DAEC_NO_ARRAY`)** builds a `libaec.a` without `aec_array.o` and `aec_res.o` (the array instance, its lanes and the stand-alone gain stage; nothing else uses them). The lane entry points of `aec.c` and the dispatch to the array are
  compiled out (`aec_array_internal.h` maps the dispatch targets to constants, so no undefined reference survives at any optimisation level). `AecConfig.n_mics` other than 1 is refused by every construction path: `aec_get_mem_size()` 0,
  `aec_init()` / `aec_init_ex()` NULL, `aec_create()` -1. `aec_get_array()`, the `aec_array_*()` diagnostics and the `aec_testing_array_*()` hooks are not defined (a call is a link error); `aec.h` is the same in both builds (documented at its top).
  The flag is part of `CFG_SIG` (and of the compile flags), so the two builds never share an `obj/` or `bin/` directory.
- **Tests.** With `AEC_NO_ARRAY=1` the array-only targets (`test-aec-array`, `test-aec-array-2`, `test-aec-array-flow`, `test-aec-res`) print a "SKIPPED" line and succeed; every other target runs, including both single-channel goldens
  (`test-split-entry*` link `aec_res.o` as test support: their reference gain stage). `test_split_entry2.c` and `test_memory_layout.c` check that `n_mics = 4` is refused; `test-header-cpp` takes the flag.
- **Size (host build, sum of the `__TEXT` sections).** `libaec.a`: 187,405 -> 165,265 B at `SIMD=1` (-22,140), 169,225 -> 150,285 B at `SIMD=0` (-18,940). A linked mono program (`aec_init`, `aec_process`, `aec_destroy`): 259,559 -> 237,303 B at `SIMD=1`
  and 235,635 -> 216,571 B at `SIMD=0`; with `-Wl,-dead_strip` 173,061 -> 166,729 B and 153,585 -> 147,373 B (a program that calls nothing else of the array loses only what `aec_init` / `aec_destroy` reach).

## [Unreleased] — 2026-11-13 — Review fixes: array beam validation, destroyed handles, status convention, header clarifications

**Output unchanged**: both golden sets (`test-split-entry`, `test-aec-array`, `-2` of each) still match bit for bit at `SIMD=1` and `SIMD=0`.

- **`aec_process_residual()` on an array validates the beam up front.** A `beamformed_error` with a non-finite bin is now `AEC_ERR_ARG` like non-finite or all-zero
  weights: no reset, the hop stays open, the previous views stay readable. It used to be `AEC_ERR_DSP` and reset all four lanes and the gain stage
  (seconds of re-adaptation for one bad bin of the application's beamformer; the Python twin already rejects non-finite beamformer output with a `ValueError` and no state change). `AEC_ERR_DSP` remains for what only shows after the arguments pass (finite weights whose projections overflow, internal
  consistency checks). Test: `short_stream` / `test_dsp_error_resets_the_instance` in `test_aec_array.c`, `test_aec_array_flow.c`.
- **A destroyed handle is deterministic.** `aec_destroy()` of an `aec_create()` instance (array or single channel) frees the arena, zeroes the caller's `Aec` and sets its `destroyed` flag;
  static (`aec_init`) instances are flagged the same way (an array's pool block keeps no state of its own for it). Every entry point then refuses the handle (`AEC_ERR_ARG`, `-1`, `0`, `NULL`, zeroed records as
  documented) instead of the heap array falling into the single-channel path and faulting. `aec_hop_size()` reads 0 for a NULL or destroyed handle, `aec_reset(NULL)` and `aec_reset()` of a destroyed handle are no-ops,
  and `aec_debug_status()` zeroes its record for a NULL or destroyed handle. A freed `Aec` or pool stays undefined (documented in `aec.h`). The flag is a byte beside `is_static` (the two were one `int`), so `sizeof(Aec)`
  (6,400 B) and every pool size are unchanged and the library keeps no global state at all; one inline helper (`aec_unusable()`) is the check at each entry point.
  Test: `test_use_after_destroy_and_null` (heap and static, array and single channel); with the flag never set, or the helper reduced to a NULL test, the test crashes (verified by mutation).
- **The array's gain stage reads the aligned far through `AecResInput.render_hop`** (the pooled `render_scaled` buffer) instead of `aec_render_stats()` into a stack array: the peak stack of
  `aec_process_residual()` on a `FULL` array drops from 8,584 to 4,472 bytes (host build, measured with a painted stack), arithmetic and results identical.
- **`aec_get_mem_breakdown()` and `aec_resolve_signal_grid()` return `AecStatus`** (`AEC_OK` = 0 on success, `AEC_ERR_ARG` otherwise) instead of `1` / `0`, so every status-returning call
  of the library reads the same way. All callers are converted: the AEC tests and CLI example, the Audio_ALG pipelines, tests and AIAEC front end, and the board samples (the "returns 1 on success" warning comments are gone);
  docs, the function map and the reference pages follow. Code that tested the result with `!` must now test `!= AEC_OK`.
- **Header clarity (`aec.h`)**: the file header no longer says "single-channel"; the aliasing rule of `aec_apply_gain()` is stated (in place on the caller's own buffer that is the view's `base_spec` is allowed
  and bit-identical to the out-of-place result, the view then holds the gained data; any other overlap is not), with a test (`test_apply_gain_in_place`); the "Array instance" section lists the config fields an array
  ignores (`delay_est_init_s`, `delay_buffer_ms`, `dt_aware_res_floor_enabled` and `ne_recent_*` for the fused gain, the 4096 ms `max_delay_ms` cap and the missing 8 kHz grid), says that
  `capture_proxy_channel` is the only microphone the shared estimator hears, states how the fused gain differs from the single-channel gain, and gives the lifetime rules of the array views
  (`error_spec` / `base_spec` alias the caller's `beamformed_error`; lane views survive `aec_reset()` uncleared). `AecDebugStatus` cites `AecLinearView.filter_converged`; "a refused call leaves the instance untouched"
  now names the `AEC_ERR_DSP` exception; comments that still named `AecLinearContext` / `AecResContext` use the current types.
- `aec_array_realign_*_lane_count` still count four soft lanes for a `changed` hop whose delta is 0 (for example a first acquisition at delay 0). No test in this repository reaches that case (skipping it
  moves no golden and no flow test), but the Python twin (`realign_soft_lane_count`) documents and counts it the same way and the `pyc_parity` run compares the pair, so the behaviour is documented in `aec.h` and kept.
- Docs/map: `docs/c_python_function_map.md` lists `aec_close_linear`, `aec_far_donor_ok`, `aec_input_finite`, `aec_unusable`, `live_array` and drops the two clock rows whose function moved into a header; `test_c_python_map.py` passes again.
- `libaec.a` `__TEXT` (host, sum of the members' `__TEXT` sections): 196,597 -> 196,661 B at `SIMD=1`, 178,669 -> 178,481 B at `SIMD=0` (this entry's net: the destroyed-handle flag, the NULL / destroyed checks and the single up-front beam test, minus the per-call `aec_render_stats` copy on the array path; the same tree with the shared destroyed control block it replaced measured 199,181 / 181,121 B, so the flag saves 2,520 / 2,640 B). Peak stack of an array `aec_process_residual()` (`FULL`): 8,584 -> 4,472 B.

## [Unreleased] — 2026-11-13 — Second golden set for the entry-point split (`test-split-entry-2`, `test-aec-array-2`)

**Output and public API unchanged.** One init-time check added: `aec_init()` / `aec_create()` of a single-channel instance fail when the carve does not end exactly at
`aec_get_mem_size()` (a lane already did); no configuration of the sweep in `test_split_entry2.c` trips it.

- New baselines written by the pre-split API (AEC 2c8ced3, audiolib 43d562a, Audio_ALG cf023eb): single-channel (22,465 lines) and array (6,903 lines), with their writers
  and sha256 (one-off files, retired on 2026-11-13 after the bit-exact comparison; see the retirement entry above).
  The first set did not reach: saturation (`saturation_level` > 0.5), double talk (`dt_indicator` > 0.2) with the near-end-recent floor, clock drift, a confirmed delay shift
  with the bad-relock revert and its reject veto, `aec_reset` mid-stream, `FIXED` / `EXTERNAL_ALIGNED` with `aec_apply_external_realign`, levels in flight,
  switched-off blocks, live observer, duty cycle, 8 kHz; for the array a clipped / silent microphone, `capture_proxy_channel` != 0, a caller-supplied `beamformed_error`,
  clock drift, `aec_set_level`, lanes without the shadow filter, the 32/36-sample admission band and quarantine moves in both directions.
- `make test-split-entry-2` (also `aec_process` alternating with the split calls, and the carve end of every instance kind) and `make test-aec-array-2` (also per-hop view invariants:
  `res_floor_protect` is the `max_dt` decision, the caller's `beamformed_error` is the fused error, the instance's gain and `aec_apply_gain` against the explicit forms) compare bit for bit
  at `SIMD=1` and `SIMD=0` on the host recorded in the READMEs; both pass with 0 mismatches. The six mutations the first set did not detect (saturated-echo min gain, clock-drift term of the HF limiter, bad-relock revert
  without the taps, `res_floor_protect` of the single-channel and of the array view, `ne_recent` hold halved) and nine more (saturation EMA attack, saturated-capture threshold,
  array saturation from lane 0 only, array clock drift forced off, DT-protect threshold, single-channel carve size, reject-list veto removed, array proxy channel ignored,
  array admission band 32 -> 33) each fail at least one of the two new tests.
- Per-hop group hashes are chained, printed every 16 hops and on every event; `G2_VERBOSE=<file>` writes the unchained per-hop hashes (the writer takes the same switch), `G2_ONLY=<substring>` selects cases.

## [Unreleased] — 2026-11-13 — Review fixes: residual getters need a completed hop, array pool and delay-ring edge cases

Both golden files (`split_entry`, `array_entry`) still compare with 0 mismatches at `SIMD=1` and `SIMD=0`.

- **Single-channel residual getters** (`aec_get_residual()`, `aec_get_residual_gain()`, `aec_apply_gain()`, `aec_apply_cng()`) return
  `AEC_ERR_SEQUENCE` / `NULL` until a residual stage has completed since construction or `aec_reset()` (they used to read an
  unwritten pool). `Aec` gains one `int`: `sizeof(Aec)` 6,392 -> 6,400; the pools do not change (both round up to 6,400).
- **Non-finite input.** `aec_process()`, `aec_process_linear()` and `aec_process_linear_shared_far_from()` on a single-channel instance return
  `AEC_ERR_ARG` for a NaN / Inf sample and change nothing, as the array already did. A refused array `aec_process_linear()` no longer
  drops the previous hop's views.
- **`aec_reset()` of a poisoned instance** now also restores `PBFDKF.erl_per_bin`; before, a finite but overflowing sample kept
  re-poisoning the first hops after the reset.
- **Array pool.** `aec_init()` accepts a pool larger than `aec_get_mem_size()` (it used to return NULL). The control block drops
  four unused members: every array pool is 256 B smaller (16k/256 FULL 724,736; 16k/512 1,132,736; 48k/1024 2,487,200).
- **Array MATCHED delay.** An estimate longer than the reference ring (`max_delay_ms` smaller than the echo path) is not accepted;
  before, `align_render()` refused it and the whole array was reset each time the estimator relocked.
- `aec_far_fft_real_compute_count(NULL)` answers 0; `aec_apply_gain_spec()` ignores NULL pointers and `n <= 0`.
- Internal: one clock shim, shared duty constants, `AEC_MAX_MICS` removed (`AEC_ARRAY_MICS`), the DT-protect threshold in one place.

## [Unreleased] — 2026-11-13 — Array lanes 1-3 reuse lane 0's far-only analysis (bit-exact, about 4 % less CPU)

**Output, public API, pool sizes (`sizeof(Aec)` 6,392, array 724,736 / 1,132,736 / 2,487,200 B FULL) and the single-channel path are unchanged.**
Both golden files (`split_entry`, `array_entry`) still compare with 0 mismatches at `SIMD=1` and `SIMD=0`.

- Lane 0 publishes what only depends on the far signal; lanes 1-3 take it instead of recomputing (`AecFarShare` in `aec_array_internal.h`, three levels):
  - `hop`, valid whenever the lanes see one far hop: reference saturation and the soft-clip it implies, the far mean square, the render-activity
    detector, the "far peak reached the floor" test.
  - `fft`, valid when the lanes' spectra agree (not on a mixed warm/soft realign hop): the far PSD, the scaled render block and the active-render
    verdict inside the post chain.
  - `analysis`, valid while the lane's far history has been identical to lane 0's since the last reset: the render-signal analyser (tonal counters,
    poor-excitation verdict, narrow-band mask) and the noise-floor update. A borrowing lane does not keep its own copy current; the array
    marks it stale, and the lane takes lane 0's state (`aec_lane_adopt_far_analysis()`) before it next computes locally.
- Identity check: after every shared-alignment change the array compares each lane's far history (`far_buffer` and the previous hop's `|X|^2` of the
  main and shadow filters, `aec_lane_far_history_equal()`) with lane 0's. A lane that differs (a soft realign clears the history, a warm one keeps it)
  leaves the shared set and follows its own analysis until `aec_reset()`; so after one mixed warm/soft realign the affected lanes stay local
  (as before this change) and only the unaffected ones keep the saving. The far history itself (`X_buf`, `x2_cache`, `far_buffer`) is not shared.
- Not hoisted: the per-lane stationarity flags and history ring (they add the lane's own reverb), the far sum of squares inside the two filters
  (13-56 ns per lane, inside the measurement noise), and everything on the near, error or weight side.
- `AecArray` flags are bytes now so the control block keeps its size; `AEC_TESTING` adds `aec_testing_array_set_far_hoist()` (forces every lane local),
  `_far_sync_lanes()`, `_far_stale_lanes()` and `_lane()`.
- Tests (`make test-aec-array-flow`, +180 checks): a hoisted array against an all-local twin over speech, tones + clipping + silence, uniform and mixed
  warm/soft realign (lane 0 on either side), the output-guard scene, a reset and a mid-stream switch, in `STATE`/`ESTIMATE`/`FULL` on the three grids.
  Outputs, the delay state and the skipped analysis state are compared bit for bit, and every lane left in sync must have an unchanged far history.
  Skipping the identity check, skipping the adopt, a wrong mask, saturation state or PSD source each fail it.
- Measured on this host (Apple M4, `SIMD=1`, per 4-lane hop `FULL`, median of 9 interleaved runs, two repetitions, one unrelated process kept a core busy
  throughout): `EXTERNAL_ALIGNED` 16k/256 24.0 -> 23.1 us (-4.0 to -4.5 %), 16k/512 43.9 -> 42.2 us (-3.9 to -4.6 %), 48k/1024 95.8 -> 91.9 us (-4.0 to -4.1 %);
  `MATCHED` (the shared estimator is common to both) -2.8 to -3.0 %, -3.2 to -3.7 %, -3.3 to -3.8 %. `libaec.a` text grows by 1.9 kB (194,025 -> 195,897 B
  at `SIMD=1`, 176,685 -> 178,493 B at `SIMD=0`).

## [Unreleased] — 2026-11-13 — The stand-alone `AecRes` API leaves the public headers

- `include/aec_res.h` is gone from the install surface: the residual-echo gain stage (`AecRes`, `AecResInput`, `aec_res_*`,
  `aec_render_stats`) is the library-internal `src/aec_res_internal.h`; the array instance and the single-channel post chain
  keep using it unchanged. Callers take the gain from `aec_get_residual_gain()` / `aec_get_residual()` and set its strength
  with `aec_set_level()`.
- `aec_cng_fill()` moves to `aec.h` and from `src/aec_res.c` to `src/aec.c`; signature, arithmetic and draw schedule are unchanged
  (the golden files still compare with 0 mismatches). Code that included `aec_res.h` for it only needs `aec.h`.
- `AecConfig.residual` comment corrected: the pool size is independent of it for a single-channel instance, and grows
  STATE < ESTIMATE < FULL for an array.
- Docs: integration §8.5 now describes `aec_cng_fill()`; the levels, usage, sequence and RES pages, the manual and the function map
  follow. Historical entries below keep the old names.


## [Unreleased] — 2026-11-13 — Array lane pools shrink by 22 % (bit-exact)

**The array pool shrinks; its output does not change, and the single-channel pool does not change.** Each of the four lanes
is now built by an internal lane path (`aec_lane_get_mem_size()` / `aec_lane_init()`, `aec_array_internal.h`) that
refuses any config but a lane's (`n_mics` 1, `EXTERNAL_ALIGNED`, residual `STATE` or `ESTIMATE`) and carves only what that
contract runs. A stand-alone `Aec`, `aec_get_mem_size()` of any single-channel config and `test_memory_layout`'s single-channel
pins (366,048 / 494,768 / 1,162,784 B) are untouched. Public API, `AEC_LAYOUT_VERSION` (1, never released) and `sizeof(Aec)` (6,392) are unchanged;
the array part of the carve text behind `build_flags_hash` now reads `fft,lane4(shared-fft,lean)`.

### Changed (array lanes only)
- **One FFT handle for the four lanes** (21,456 / 40,656 / 79,056 B). The handle holds the twiddle config and a staging block
  that every call overwrites before reading; lanes run strictly one after another within a hop (the main filter, the shadow
  filter and the post chain of one lane already shared one handle). It sits in the array pool and is counted in `fusion_bytes`.
- **No delay-shift snapshots** (59,968 / 69,952 / 205,248 B): `delay_shift_judgment_pre_W`, `_pre_shadow_W` and `_pre_H_error` are
  written and read only inside the MATCHED estimator's path (`has_delay`), which a lane never has.
- **No synthesis or gain buffers** (45,952 / 86,912 / 181,120 B for the four lanes): `ola`, `eout`, `eout_full`
  (`aec_post_apply_output()` runs only for `FULL` without `context_only`), the eleven SuppressionGain arrays (only
  `suppression_gain_get_gain()` reads them; a lane tracks `DominantNearend` only), the REE render-history rows (read only
  without the echo-generating window) and `x2_past` (read only when the filter has no `x2_cache`). `aec_post_reset()` and
  `suppression_gain_init()` accept NULL for these; `aec_post_apply_output()` and `suppression_gain_get_gain_stats()` assert.
- **The lane's shadow filter follows the main filter** (55,872 / 74,112 / 197,120 B): `pbfdaf_init_static_follower()` shares
  the main filter's `X_buf` and `x2_cache` and drops `near_spec`, `error_spec_windowed` and `spec_scratch` (its frontend is
  lightweight). Sharing is exact only because every writer of the far history is paired between the two filters: both run every hop with the same far
  input (the main filter takes the shadow's far spectrum), both advance `partition_idx` on every path, and every
  `pbfdkf_reset` / `pbfdaf_reset` / `reset_taps` / `clear_far_history` / `warm_shift_ir` call site in `aec.c` hits both. An
  instrumented build of the pre-change code compared the two filters' `X_buf`, `x2_cache` and `partition_idx` before the shadow and after the
  main filter on every hop of every test that runs a shadow (1,172,330 comparisons, 0 differences, mixed realign and reset included).
- **Stationarity history rows** (20,608 / 53,440 / 90,240 B): a lane carves the ring's `aec_blocks_to_hops(13, ...)` rows
  (`stationarity_estimator_window_hops()`) instead of 16.
- The lane carve is checked to end exactly at the size it announced.

### Result (host, balanced, `MATCHED`, `delay_num_filters = 5`; bytes)
| grid | STATE | ESTIMATE | FULL | before (FULL) | lane each (before) |
|---|---:|---:|---:|---:|---:|
| 16k/256 | 710,576 | 716,832 | 724,736 | 928,848 | 148,288 (201,040) |
| 16k/512 | 1,106,288 | 1,118,688 | 1,132,736 | 1,458,064 | 245,104 (329,760) |
| 48k/1024 | 2,433,104 | 2,457,792 | 2,487,200 | 3,240,240 | 539,472 (734,256) |

The array is 204,112 / 325,328 / 753,040 B smaller (22.0 / 22.3 / 23.2 %) for all three residual modes. `lane_bytes`: 593,152 /
980,416 / 2,157,888; `fusion_bytes` (FULL) 24,672 / 44,384 / 86,880 (it now includes the shared FFT handle: 7,152 / 13,552 /
26,352). `libaec.a` `__TEXT` (`size`, sum over members): SIMD=1 191,730 -> 194,025 B, SIMD=0 174,422 -> 176,685 B (+2.3 kB: the lane path and the follower filter variant).

### Verified
- `make test-split-entry` and `make test-aec-array` (both golden files, not regenerated) bit-exact at SIMD=1 and SIMD=0
  (76,506 and 263,031 compared lines, 0 mismatches), `test-aec-array-flow`, `test-memory-layout` (now also pins the array pools and
  walks 216 array configurations: three grids x STATE / ESTIMATE / FULL x three delay modes x shadow / live observer / mic high-pass
  on and off, each refused at `need - 1`, accepted at `need` with the tail canary intact, then 60 hops, `aec_reset()` and 10 hops),
  and the rest of the AEC `test-*` targets. Mutation checks on a scratch copy: a shadow `X_buf` offset by one element fails
  `test-aec-array` (125,491 mismatching golden lines); one fewer stationarity-history row fails it (61,820).
- Not done: a `STATE` lane still carves the residual-echo estimator arrays and the comfort-noise arrays its early exit never
  reads (about 26 kB per lane at 16k/256), because the estimator's scalars are read before that exit and its init / reset write
  the arrays; the embedded `DelayAec` (1,488 B) in `Aec`; the far-side state per lane (see below).

## [Unreleased] — 2026-11-12 — Four-microphone array instance (`AecConfig.n_mics = 4`)

**New C API surface; the single-channel instance is unchanged.** The four-microphone linear flow that Audio_ALG's 4ch `mcab`
kept over four caller-owned lanes, a caller-owned delay estimator and a stand-alone `AecRes` is now one instance of this
library: `AecConfig.n_mics = AEC_ARRAY_MICS` and one `aec_init()` / `aec_create()` place four lanes (each a single-channel
`EXTERNAL_ALIGNED` instance), one shared delay estimator with one aligned far, and the fusion buffers in one pool. The
beamformer (SRP-PHAT, BF, VAD) and the synthesis stay with the caller.

### Added
- Config: `n_mics` (1 or 4), `capture_proxy_channel` ([0, 3]), `delay_backward_quarantine_enabled` / `_s`. An array instance
  accepts 16k/256, 16k/512 and 48k/1024 only, `residual` STATE / ESTIMATE / FULL (not OFF) and `max_delay_ms` <= 4096;
  `delay_est_init_s` and `delay_buffer_ms` are not read by it.
- Per hop: `aec_process_linear(a, mics, ref)` (interleaved `float[hop][4]`) -> `aec_get_array()` (`AecArrayView`: each lane's
  analysis spectrum, rendering base, formed hop, guard flag, the aligned far and the shared `AecDelayState`) -> the caller's
  beamformer -> `aec_process_residual(a, &beam)` (`AecBeam`: channel-major weights `Complex[4][n_freqs]`, optional
  `beamformed_error`) -> `aec_get_residual_gain()` / `aec_apply_gain()` / `aec_apply_cng()`.
- `AecStatus` gains `AEC_ERR_DSP` (-4): a lane or the fusion refused the hop; the whole instance has been reset as by `aec_reset()`.
- `AecMemBreakdown` gains `lane_bytes` and `fusion_bytes`; diagnostics `aec_array_*` (warm / soft realign lane counts, pending
  delay candidate, duty census, matched-filter and lane counts, `aec_array_get_timing()`).
- Mode behaviour: with `STATE` the residual stage is `AEC_ERR_MODE` (lanes only); with `ESTIMATE` it fuses without a gain and may
  be repeated on a hop; with `FULL` a second `aec_process_residual()` on the same hop is `AEC_ERR_SEQUENCE`. In the fused
  `AecResidualView` `formed_hop` and `echo_spec` are `NULL`. `aec_process()`, `aec_process_linear_shared_far_from()` and
  `aec_apply_external_realign()` are refused on an array instance (`AEC_ERR_MODE`, `AEC_ERR_MODE`, `-1`).
- `aec_post_reset_serial()` of an array is the sum of the lanes' counters: `aec_reset()` moves it by 4.
- Tests: `make test-aec-array` (every line of the baseline written by the pre-fold `mcab` flow, a one-off file retired on 2026-11-13, plus contract checks)
  and `make test-aec-array-flow` (524 state-machine checks ported from `test_mcab.c`; the map of what moved and what stays in Audio_ALG was in the retired
  `PORTING.md`). With SIMD=1 on the freeze host (Apple arm64,
  Apple clang 17, NE10): 138,315 golden lines consumed, 0 mismatches, 612,258 checks, 0 failed; flow tests all passed. SIMD=0 was not re-run for this entry.
- Docs: `docs/reference/{integration,lifecycle,configuration}.md` (section 8.8, array pool, section 6.8), `docs/c_user_manual_zh_TW.md`,
  the four HTML reference pages, `c_impl/README.md`, `c_impl/STATIC_MEMORY.md`. The C example in 8.8 compiles with
  `-Wall -Wextra -Werror` and runs.

### Array pool sizes (host measurement; balanced, `MATCHED`, `delay_num_filters = 5`)
| grid | STATE | ESTIMATE | FULL | `lane_bytes` | `fusion_bytes` (FULL) |
|---|---:|---:|---:|---:|---:|
| 16k/256 | 914,688 | 920,944 | 928,848 | 804,160 | 17,776 |
| 16k/512 | 1,431,616 | 1,444,016 | 1,458,064 | 1,319,040 | 31,088 |
| 48k/1024 | 3,186,144 | 3,210,832 | 3,240,240 | 2,937,024 | 60,784 |

Unlike a single-channel instance, the array pool depends on `residual`. The `Aec` header (6,400 B) is in none of the breakdown fields.

### Not done
- The far-side state is still held per lane; only the delay estimator, the aligned far and, within a hop, lane 0's far FFT
  result (borrowed by lanes 1-3) are shared.
- Each lane still carves its full single-channel `EXTERNAL_ALIGNED` pool (201,040 / 329,760 / 734,256 B, identical for STATE and
  ESTIMATE lanes), including storage its mode does not use; about 87 / 90 / 91 % of the `FULL` array pool. Reducing it is a later step
  (done in the 2026-11-13 entry above; the table above is the pool before it).
- Audio_ALG's `mcab` and the stand-alone `AecRes` path are not yet moved onto the array instance.
- The pre-split baseline files (retired on 2026-11-13) were produced and compared on one host only; other hosts and the target board are unverified.


## [Unreleased] — 2026-11-12 — Split entry points and `AecConfig.residual` replace the context API

**Breaking C API change; output bits unchanged.** The single-channel instance now has three entry points and one setting
for how far the residual stage goes. Every old-API configuration maps onto a new mode (old-to-new list below), and the per-hop
outputs of the new calls equal the old ones bit for bit: `make test-split-entry` reproduces all 76,506 compared lines of
the one-off baseline files of the pre-split API (Apple arm64 / Apple clang 17, NE10; both SIMD settings; retired on
2026-11-13 after a re-run of the old API reproduced them byte for byte, see the retirement entry above).

### Changed (API, breaking)
- **`AecConfig.residual`** (`AecResidualMode`: `AEC_RESIDUAL_OFF`, `STATE`, `ESTIMATE`, `FULL`; default `FULL`) replaces the
  three flags `enable_res`, `return_res_context` and `spatial_linear_context` (and the `AEC_CONTEXT_*` constants). Old to new:
  `enable_res = 0, return_res_context = 0` is `OFF`; `AEC_CONTEXT_LINEAR_ONLY` is `STATE`; `AEC_CONTEXT_NO_GAIN` is `ESTIMATE`;
  `enable_res = 0, return_res_context = 1` is `FULL` driven by `aec_process_linear()` + `aec_process_residual()`;
  `enable_res = 1` is `FULL` driven by `aec_process()` (the library applies the gain, adds comfort noise and synthesizes).
  Validation is one range check on `residual`; the old flag-combination rules are gone.
- **Entry points return `AecStatus`** (`AEC_OK` = 0, `AEC_ERR_ARG`, `AEC_ERR_MODE`, `AEC_ERR_SEQUENCE`, `AEC_ERR_DSP`) instead of
  `void`: `aec_process(a, mic, ref, out)`, `aec_process_linear(a, mic, ref)`, `aec_process_residual(a, beam)` (single channel: `NULL`) and
  `aec_process_linear_shared_far_from()`. A refused call touches nothing. `aec_process_context()` and
  `aec_process_context_shared_far_from()` are removed.
- **Views**: `aec_get_linear()` / `AecLinearView` (linear stage and its status scalars) and `aec_get_residual()` /
  `AecResidualView` (formed hop, analysis spectra, `base_spec` and `guard_active`, `r2`, `comfort_noise`, `res_gain`, the flags the gain
  stage was fed) replace `aec_get_linear_context()`, `aec_get_res_context()` and `aec_get_output_base()` (and `AecLinearContext`,
  `AecResContext`, `AecOutputBase`). `struct_size` is validated as before. `aec_get_residual_gain()` returns the `G_res` buffer.
- **Gain helpers**: `aec_apply_gain()` and `aec_apply_cng()` (generator inside the instance, reseeded by every hard post-chain reset),
  the stateless `aec_apply_gain_spec()` / `aec_apply_cng_spec()`, and `AEC_CNG_NR_GAIN_FLOOR`. `AEC_OUTPUT_GUARD_API_VERSION` is now 2.
  Migration steps: `docs/sdk_migration_output_guard.md`.
- A hop may alternate between `aec_process()` and the split pair on one instance. The filter and detector state advance identically;
  the library's own overlap-add tail and comfort-noise generator advance only in `aec_process()` and `aec_apply_cng()`.
- `sizeof(Aec)` grows from 6352 to 6392 bytes and every `aec_get_mem_size()` by 48 bytes (the struct is carved 16-byte aligned) (balanced, default delay mode:
  16k/256 366,048; 16k/512 494,768; 48k/1024 1,162,784; without the shadow filter 323,168 / 434,080 / 1,008,848).
  `sizeof(AecConfig)` is 232 bytes.

### Tests and docs
- New `test/test_split_entry.c` (`make test-split-entry`, in the `make test` tier).
- Every single-channel consumer in `c_impl/test`, `example/aec_wav.c` and `test/pyc_parity/aec_run.c` was converted. Where a test
  compared `aec_process_context()` with `aec_process()` it now compares the split pair with `aec_process()` (views, byte for byte);
  the flag-combination checks of `test_config_validation.c` became a range check plus the per-mode exposure of the views;
  `test_guard_unify.c` gained a second split-driven lane rendered with `aec_apply_gain()` / `aec_apply_cng()` and compares it with
  `aec_process()` bit for bit. `test/pyc_parity/aec_run.c` still takes the Python harness's `--no-res` / `--ctx`
  switches and writes byte-identical output files (60 flag/grid combinations compared against a build without this change; only the stderr text of the `--base` error path differs).
  `test_memory_layout.c` pins the new sizes.
- C rows of `docs/c_python_function_map.md`, the reference pages and the HTML pages follow the new names.


## [Unreleased] — 2026-11-02 — Delay matched filter and tests follow the shared 32-bit ARM NEON gate

### Changed
- `delay_aec.c`: the matched-filter NEON body is keyed on `SK_HAVE_NEON` (AArch64, and 32-bit ARM with NEON plus fused FMA) instead of
  a private 32-bit special case; the across-lane add is the shared `vaddvq_f32`, so `da_vaddv_f32` is gone. The private `DA_MF_NEON` and
  `DA_DISABLE_ARM32_NEON` switches are removed: `SIMD=0` / `SIMD_KERNELS_FORCE_SCALAR` is the one scalar switch. Output is unchanged on AArch64; on 32-bit ARM it is bit-identical for finite, non-subnormal inputs (the vector
  unit flushes subnormals).
- Tests that compare a NEON kernel with its scalar twin (`simd_selftest_aec`, `test_neon_twins`, `test_delay_kernels`) call
  `SK_MATCH_SCALAR_TO_VECTOR_MODE()` first; `test_guard_unify` no longer assumes `sizeof(size_t) == 8` in its short-struct case;
  `test_delay_matched_neon` names the 32-bit ARM body correctly.

## [Unreleased] — 2026-10-28 — output guard unified across API modes

The output guard (`output_capture_when_linear_unusable`, always on in the shipped levels), the comfort-noise generator
and the post-chain reset now behave the same through `aec_process()`, `aec_process_context()` and the lean
`spatial_linear_context` modes, in C and in Python. One algorithm; splitting the API no longer changes state or results.
The same entry ports Python's bad-relock soft revert to C, a deliberate C behaviour change on the default path (below).

### Changed (behaviour)
- **Context-only configs no longer substitute the capture into the analysis path.** With `enable_res = 0` and
  `return_res_context = 1` the FORM step used to admit the capture as a third candidate when the previous hop's usable
  verdict was false and the selected residual carried more hop energy than the capture. That candidate
  (`_SEL_CAPTURE` / `LFS_SEL_CAPTURE`), its 30-sample crossfade and its previous-hop verdict are gone.
  `formed_hop` / `formed_output`, `error_spec`, `echo_spec` and `near_spec` are the real selected linear residual in every
  mode, and AecState, ERLE, R2 and the error and echo PSDs always see it. `error_spec` is the STFT of
  `[previous formed hop | current formed hop]` on every hop, guard hops included, so a unity IFFT with overlap-add
  reproduces the previous formed hop one framing hop late. `AecLinearContext.formed_linear_hop` follows. Because the
  adaptive state now sees the residual where it used to see the capture, a context-only output that differs on such a hop
  can keep differing on the hops after it; the change is not confined to the hop where the substitution used to fire.
  The same holds for the guard, the reset serial and the soft revert below: a difference that starts at an event can
  persist after it (the adaptive state, the OLA tail and the comfort-noise sequence carry it forward).
- **The guard now only selects the rendering base, once per hop, in every mode.** It is evaluated after this hop's
  usable verdict is final and before any context or linear-only exit: with the guard enabled, the verdict false and
  `sum(|E|^2) > sum(|Y|^2)` (strict, windowed spectra, `Y = E + echo`, the same float32 pairwise reduction in C and
  Python), the base is `Y` and `guard_active` is set; otherwise it is `E`. Full mode renders `base * gain` as before.
  Identical spectra give the identical decision in both languages. The two STFTs are computed separately and their
  upstream spectra differ by about 1e-7 relative, so the C-versus-Python contract is a tie band, not bit equality (see
  "C-versus-Python parity contract" below).
- **Rendering base exposed.** C: `aec_get_output_base(const Aec*, AecOutputBase*)` with
  `AecOutputBase { struct_size, n_freqs, guard_active, base_spec }` (`struct_size` is an input and is validated before
  anything is written: a value below `sizeof(AecOutputBase)` returns -1 and leaves the caller's struct untouched, a larger
  value, e.g. from a newer caller, is accepted and only the known prefix is written, with `struct_size` coming back as
  `sizeof(AecOutputBase)`; `base_spec` aliases a dedicated buffer that comfort noise and other scratch never touch, valid until the next
  processing or reset call). Python: `AecResContext.output_base_spec` and `guard_active`. Consumers render
  `output_base_spec * gain`, never `error_spec`. The base does not pair with `formed_hop`. `AecResContext` itself gained no
  field in C.
- **Comfort noise: one seed, one numeric core.** `AEC_CNG_SEED` (42) is exported (`aec_post.h`, `modules/cng.py`) and is
  the seed after construction and after every hard post-chain reset. `aec_cng_fill()` is now the only implementation:
  the output stage of `aec_process()` calls it with its total gain as `g_res` and `g_nr = NULL`, so the public helper is
  bit-identical to the in-chain noise on the same inputs. The amplitude is
  `(sqrt(1 - g_res^2) * max(g_nr, nr_floor)) * (sqrt(cn / 32768^2) * table entry)` per component, one LCG draw per bin
  `1 .. n-2` in ascending order. The previous public helper evaluated `n_amp * gain * nr * table` with libm `sqrtf` and a
  division, so its results can differ from the old ones in float rounding. Python gains `modules/cng.py` (`cng_fill`,
  `cng_draw_indices`, `cng_noise_terms`) shared with the in-chain noise. The seed itself is not new for the library: the
  in-chain comfort noise of the full AEC already started from 42 in both languages, so full-mode output does not move
  because of the seed. Only external consumers that seeded their own comfort noise with the previous `0x9e3779b9` see a
  different noise sequence, and only on hops where that noise is actually added.
- **Reset serial.** C `uint32_t aec_post_reset_serial(const Aec*)`, Python `AEC.get_post_reset_serial()` and
  `AecResContext.post_reset_serial`: a non-consuming, wrapping counter of hard post-chain resets. There are two sources.
  `aec_reset()` is a separate call, not part of any processing call: the serial already reads +1 when it returns, and the
  next processing call starts from the cleared state. Inside a processing call only a confirmed delay change that restarts
  the filter counts (the hard branch of a first acquisition, or of a confirmed shift), done before that hop's synthesis.
  Each counted reset clears the library's OLA tail and reseeds its CNG to `AEC_CNG_SEED`. Warm-transfer and soft
  realignments, the bad-relock revert (soft, see below: OLA tail and CNG keep running),
  `aec_apply_external_realign` and `AecLinearContext.generation` changes do not count. An application that synthesizes
  outside the library keeps its own last-handled serial; when the value after a processing call (or after `aec_reset()`)
  differs it clears its OLA tail and reseeds its CNG before its first CNG draw and before the inverse-transform push of the
  hop it renders from that call (after `aec_reset()`: the first hop processed afterwards), and never resets the AEC in
  response.
  The library touches neither NR nor NN state; following the reset there is the application's policy.
- **Synthesis window accessor.** C `int aec_synthesis_window(const Aec*, const float** window, int* length)` returns the
  window the library applies before overlap-add (block_size entries, valid for the instance lifetime). An external
  synthesizer borrows it through `AcStftConfig.window` for its `AcIstft`; the audiolib default sqrt-Hann differs from it in
  the last float bit, so bit-exact full-vs-split needs this table. The pointer is read-only and owned by the instance;
  `*window` and `*length` are written only on success (-1 for a NULL instance, window or length). No behaviour change.
  `make test-synthesis-window`.
- **Bad-relock revert is now a soft revert in C (deliberate C behaviour change on the default path).** After a
  confirmed delay shift the judge compares windowed ERLE 100 hops later; a relock more than 1 dB worse than the pre-shift
  value is reverted and its delay remembered as rejected. Python has done this as `_apply_delay_shift_soft_revert` since
  `9e1bea4` (2026-10-18), restoring the pre-shift taps without the hard-reset chain; C was last synchronised at `fb84a2c`
  (both used the hard-reset chain then) and kept the hard reset, so C and Python separated at every revert. C now has
  `aec_apply_delay_shift_soft_revert()`: it restores the pre-shift delay, the main W and `H_error_per_bin` (a new
  `n_freqs` float snapshot taken at the shift) and the shadow W, clears the far history only when the revert retards the
  alignment, and resets the windowed-ERLE accumulators, the main/shadow error EMAs and the pending delay. It does not run
  the hard-reset chain, so `aec_post_reset_serial()` does not move and the OLA tail and CNG keep running; the delay
  generation still bumps, so seam consumers see the change. **Effect on C output:** it differs from the previous C
  output only on clips that reach the revert, from the revert hop on, and there C moves onto Python's output: 16 of 800
  blind clips at 16 kHz/256 (17 events) and 15 of 800 at 16 kHz/512 (16 events), none in the near-end-only set (round-7
  scan of the 115e10c tree), and 11 of 300 at 48 kHz/1024 (counted by the port step; not re-scanned in this documentation
  pass). These are counts on these blind sets, not a fixed fraction of clips or of relocks: how many clips revert
  depends on the material. The delay-event stream (every acquisition, shift and revert, same hop, same value) is unchanged on all 800
  clips at both 16 kHz grids; without a revert the output is bit-identical to the previous C. Before the port, after a
  revert the two languages differed by -28 to +2 dB of the output level, with peaks up to full scale; afterwards they
  stay at the pre-event float32 level on 30 of 31 measured clip-grid cells. Python is unchanged.

### Unchanged
- **Full-mode `aec_process()` output is bit-identical to 115e10c** in C, with one deliberate exception: the output hash
  over every clip of the blind sets matched at 16 kHz/256 (800 clips), 16 kHz/512 (800) and 48 kHz/1024 (300), 0 differing
  clips each, measured before the bad-relock soft-revert port above. The port changes the output of the clips that reach
  a revert (counts above) and nothing else. The Python full-mode output matches 115e10c on the 60-cell subset checked
  (0 differing cells); Python is untouched by the port, and the behaviour hash recorded under Compatibility (`9963b6d7...`) was re-computed on the final tree.
- Split rendering (`aec_process_context()` + `aec_get_res_context()` + `aec_get_output_base()` + `aec_cng_fill()` +
  audiolib `ac_istft_push()` with the library's synthesis window, NR/NN off) is bit-identical to `aec_process()` in the same
  build (`make test-guard-unify`: synthetic streams covering init, never usable, unusable to usable, a moving echo path with
  delay changes and hard resets, guard switching on and off, `aec_apply_external_realign`, `aec_reset` and replay, plus the
  blind-set clips of `guard_unify_clips.tsv`, never-usable clips included; the revert scenarios check that a revert leaves
  the serial unchanged and does not reseed the full-mode CNG, and that a renderer that clears and reseeds on the revert
  hop would produce different audio). The real-clip part reports SKIP loudly when the gitignored corpus is absent; the
  synthetic scenarios always run.

### Guard statistics
- 97.72 % of the guard-firing hops at 16 kHz/512 (64,096 hops over 800 clips) occur before the first usable verdict
  (97.43 % of 118,078 at 16 kHz/256, 99.46 % of 137,397 at 48 kHz/1024). Some clips never get a usable verdict (228 of
  800, 233 of 800 and 88 of 300 clips at the three grids) and the guard fires on 72, 74 and 69 of them, so firing is not
  confined to the first seconds. Tests, training data and quality checks must include never-usable clips.

### Compatibility
- **Static pool grows** by one `n_freqs` Complex slice (the dedicated rendering-base buffer), one `n_freqs` float slice (the
  main-filter `H_error` snapshot of the bad-relock soft revert) and the 16-byte-aligned growth of the `Aec` struct
  (`sizeof(Aec)` 6336 to 6352 bytes). Final `aec_get_mem_size()` at the balanced level, default delay mode (NE10, LP64),
  against 115e10c:

  | grid | shadow on, before | shadow on, now | shadow off, before | shadow off, now |
  |---|---:|---:|---:|---:|
  | 16 kHz/256 | 364,416 | 366,000 (+1,584) | 321,536 | 323,120 (+1,584) |
  | 16 kHz/512 | 491,600 | 494,720 (+3,120) | 430,912 | 434,032 (+3,120) |
  | 48 kHz/1024 | 1,156,544 | 1,162,736 (+6,192) | 1,002,608 | 1,008,800 (+6,192) |

  The growth is `align16(K * 8) + align16(K * 4) + 16` with `K = fft_size / 2 + 1` (1,040 + 528 + 16, 2,064 + 1,040 + 16,
  4,112 + 2,064 + 16). Re-query `aec_get_mem_size()` and re-init; a pool sized for the previous build is refused by
  `aec_init`. `test_memory_layout.c` pins `sizeof(Aec)` and the six pool sizes above as literals (skipped on a non-LP64
  ABI) in addition to the structural check against the recorded 115e10c sizes. `docs/c_user_manual_zh_TW_01_lifecycle.md`
  (all pool tables), `docs/html/aec.html`, `aec_c_api.html`, `aec_params.html`, `STATIC_MEMORY.md` and the
  `--print-mem-size` example carry the new values, measured once on the final tree with the same API (`aec_get_mem_size`).
  The `AecRes` pool and the live-observer increments are unchanged (7,904 / 14,048 / 29,408 B and +38,768 / +58,656 /
  +141,648 B at the three grids).
- Internal headers: `linear_filter_select()` lost its `linear_unusable` and `capture_fallback_enabled` parameters and
  `LFS_SEL_CAPTURE` is removed; `aec_post_apply_output()` takes the rendering base instead of the error, echo, magnitudes
  and verdict; `aec_post_init()` lost `sqrt2_sin_lut`; `AecPostAbs` lost `abs_ybase`; `AecPost` gained `reset_serial` and
  `output_guard_active`; `AecPostRunScratch` gained `output_base`; `aec_post_eval_output_guard()` is new.
- **Python behaviour hash** (`aec_python_behavior_hash()`, AIAEC): `9fa7a2de...` to `9963b6d7...`. This is an
  output-changing change and is not registered as a byte-equivalent migration. **AIAEC consequence:** the AIAEC front end
  reads the analysis seam, which in context-only configs was previously the capture on guard hops and is now the real
  residual, so `linear_error` changes. The corpus is regenerated and the model retrained by the owner; checkpoints and the
  released inference-only pointer built at the previous hash are retired or kept as a legacy front end, never silently
  re-pointed. Three AIAEC guards go red at the new hash:
  `test_released_checkpoint_exception_is_separate_and_targets_this_build`,
  `test_released_checkpoint_exception_is_inference_only_and_warns` and
  `test_inference_loads_the_released_pre_constraint_checkpoint_with_warning`. The analysis residual may exceed the capture on
  a guard hop; the final audio is protected by the rendering guard.
- Required change in consumers that render audio from the context (mono, DFN and ULCNet lanes, 4ch lanes, board samples):
  render `output_base_spec * gain` instead of `error_spec * gain`, handle the reset serial, and seed their comfort noise
  with `AEC_CNG_SEED`. Each consumer states which spectrum feeds its NR estimator. Defaults are unchanged (the library
  CNG default is not touched; CNG in board samples stays a separate `--cng` configuration).
- **Breaking behaviour for existing consumers.** A frequency-domain consumer that still renders `error_spec` instead of the
  rendering base silently loses the guard: on guard hops its output carries the residual that is louder than the capture
  (nothing fails at compile or run time). Time-domain seam consumers (`formed_hop` / `formed_output`,
  `AecLinearContext.formed_linear_hop`) are the analysis residual on every hop and have no rendering base, so they are
  unguarded by construction. A consumer that plays or re-synthesizes them gets no output guard. Migration steps for SDK
  applications: `docs/sdk_migration_output_guard.md`.
- **Bad-relock revert and the serial.** A consumer written against the earlier C, where a revert bumped
  `aec_post_reset_serial()` and cleared the library's OLA tail, no longer sees either on a revert; its own OLA and CNG
  handling needs no change as long as it follows the serial.

### Added
- `c_impl/test/test_guard_unify.c` (`make test-guard-unify`, in the SIMD-dependent group) with
  `c_impl/test/guard_unify_clips.tsv`; `python/tests/test_guard_unify.py` with `python/tests/guard_scenes.py` (ports of the C
  relock and soft-realign streams; the serial must move +1 on hard events and 0 on every other delay event, a revert must not
  reseed the CNG, and a renderer that reseeds on the revert hop must produce different audio); a guard threshold fixture at
  equal energy and plus or minus one ulp in both languages; absolute pins of `sizeof(Aec)` and the six pool sizes in
  `test_memory_layout.c`.
- `c_impl/test/pyc_parity`: 16 guard cells (8 guard cells plus 8 `guard.revert.{Y91u,IUp2,rOza,waxU}.{16k256,16k512}.full`
  cells on blind clips that revert), per-hop current delay dumped by both drivers (`<pre>.dly.txt`), exact gates for the
  delay, the revert hops and a minimum number of reverts. `inputs.sha256` gained the four revert clips; no existing hash
  changed.
- `docs/c_python_function_map.md`: rows for `aec_get_output_base`, `aec_post_reset_serial`, `aec_post_eval_output_guard`,
  `aec_synthesis_window`, the `modules/cng.py` helpers, `AEC._post_output_base`, `AEC._guard_energy` and
  `AEC.get_post_reset_serial`, and `aec_apply_delay_shift_soft_revert` <-> `AEC._apply_delay_shift_soft_revert` as a
  `mirror` row (the earlier "Python-only" row and its D-15 citation, which belongs to the live-observer lane, are gone;
  D-15 now says only the observer taps are not restored by the revert, in either language); the stale
  `AEC._post_output_spectrum` row is removed. The sequence and RES flow diagrams (`aec_sequence.html`, `aec_res.html`) now
  render from `output_base_spec * gain` and include the serial step.

### C-versus-Python parity contract (pyc parity, guard cells)
- **Tie band.** C and Python output-guard decisions must match on every hop except inside a tie band of 1e-6 relative
  energy margin, `|Σ|E|² − Σ|Y|²| / Σ|E|²`, on hops that are unusable on both sides (a usable-verdict difference is never a
  tie). The band is 3.3 times the largest measured near-threshold C-versus-Python margin disagreement (3.3e-7 at
  48 kHz/1024; 3.0e-7 at 16 kHz/256) and 4 times the largest margin at any observed mismatch (2.48e-7). Hops inside the
  band are counted and printed per cell; any mismatch outside the band fails the cell. The fixed near-threshold fixtures
  (identical inputs) keep exact equality.
- **Result on the 16 guard cells** (`--full`): no guard mismatch outside the band (81, 81 and 27 hops inside the band on the
  two 16 kHz never-usable clips, 0 elsewhere); `post_reset_serial` identical on every hop (hard-reset cells 4/4, 4/4 and 3/3
  C/Python changes; the earlier 5/4, 5/4 and 5/3 divergence is gone after the soft-revert port); current delay identical on
  every hop; bad-relock reverts at the same hop in both languages on all 12 revert-bearing cells.
- **Green:** `never_usable.16k256` lean and full; `hard_reset` 16k256, 16k512 and 48k1024 lean and full (these include reverts
  at hops 553, 327 and 858 + 1038); `revert.IUp2.16k512`; and, under the `res_gain` gate below, `revert.Y91u` (both grids),
  `revert.rOza.16k256` and `revert.waxU.16k256`. **INFO(T-01)**: `never_usable.16k512`, `never_usable.48k1024`,
  `revert.IUp2.16k256`, `revert.rOza.16k512` and `revert.waxU.16k512`. No cell fails a hard seam gate as run.
- **Registered float32 ties (T-01, INFO only): a known numeric divergence, not an equivalence.** As run, C and Python do
  not agree in these cells: the analysis-seam context (ERLE factor, error or echo spectrum) is over tolerance in all five
  and `res_gain` in two (`revert.rOza.16k512`, `revert.waxU.16k512`), and the raw outputs can exceed
  the Layer-1 audio tolerance (`revert.rOza.16k512`: out max_abs about 0.0527, 176 times the 3e-4 tolerance, `res_gain` up
  to 0.96 apart; the other four stay under the audio tolerance, 1.0e-6 to 2.2e-4). That the difference comes from the
  registered float32 tie is shown only by the forced-decision experiment below; nothing shows that the unforced C and
  Python outputs are the same. In those five cells one or a few per-bin hard decisions inside the
  adaptive filter sit within the C-versus-Python rounding difference of their own operands: the refined filter's H_error
  leakage compare `e2_refined[k] <= e2_coarse[k]` (all five), and in `revert.rOza.16k512` also the filter noise gate
  `X2[k] >= gate` (hop 543 bin 148, margins about ±1e-7). C takes one side and Python the other, and the one-bin
  difference spreads through the adaptive state to the analysis seam and the audio. Each cell registers the decision(s)
  as hop:bin:side; the Python parity driver gained diagnostic `--force-leak H:K:C[,...]` and `--force-gate H:REL[,...]`
  options to take them the way C does. The cell is INFO only when (1) the hard seam gates (guard_active outside the tie band,
  post_reset_serial, delay events and reverts) are green as run, and (2) the Python run with the registered decisions forced
  is green on every audio, context and guard-seam gate. It fails if either stops holding, and it fails once the unforced cell
  is green (remove the marker). Forced runs: out max_abs 1.6e-7, 5.2e-8, 2.8e-7, 3.9e-7 and 1.06e-6. If the registered
side in `never_usable.16k512` is flipped to Python's own choice, the cell fails ("T-01 shape lost"). The marker sits in
  the `pyc_parity.py` registry next to N-01 and N-02; no expected-red entry was added or enlarged.
- **`res_gain` context gates (a new acceptance contract, not the old one tightened).** The earlier gate was the per-bin,
  per-hop maximum `CTX_gain_max_abs` (1e-3); it is now printed only. That maximum is reached in the quietest bins (75 to
  85 dB below the hop peak, where float32 rounding is about 1e-3 of the bin) and, through the max-gain ratchet, which
  multiplies an earlier hop's difference by 2 or 4 in amplitude, in loud bins too. Green cells already reach 9.96e-4
  (`hard_reset.16k512.lean`). `res_gain` is now held by three gates, all in `pyc_parity.py`:
  - relative L2 weighted by each bin's near-end power `|E + echo|²` (`CTX_gain_wrel_l2` = 1e-4). At most 1.52e-5
    (`revert.IUp2.16k256`, against its 1-ulp branch) across the 16 guard cells. It is a whole-cell figure, so on its own it
    dilutes an error confined to a few hops or bins;
  - max |C − Python| over the bins within 20 dB of the hop's peak near-end power (`CTX_gain_loud_max_abs` = 3e-3), every
    context cell. Worst green 1.17e-3 (`revert.rOza.16k256`), margin 2.6 times;
  - the spectrum an application renders, `output_base_spec * res_gain`: per hop `‖Pc − Pp‖ / ‖Pp‖`, maximum over the hops
    within 60 dB of the cell's loudest product hop (`CTX_render_hop_rel_l2` = 3e-3). Only the 16 guard cells dump the
    base, so the 3 other context cells have the first two gates only. Worst green 1.06e-3 (`revert.waxU.16k256`), margin
    2.8 times; per-hop p99 at most 5.6e-5 and whole-cell at most 1.0e-5 (printed).

  The forced runs of the five T-01 cells read at most 5.3e-4 (loud bins) and 1.9e-4 (render product). Unforced,
  `revert.rOza.16k512` (9.6e-2 weighted) and `revert.waxU.16k512` (2.0e-4) run on the other side of their registered tie and
  are over all three gates; both are green once the tie is forced. A planted +0.1 % change of the suppression threshold
  `enr_suppress` (Python side) fails all three gates on `revert.Y91u.16k256`, `revert.waxU.16k256` and
  `hard_reset.16k256.lean`: weighted 3.0e-3, 1.3e-3, 7.4e-4; loud bins 0.055, 0.25, 0.058; render product 0.16, 0.12, 0.066.
  An error the weighted figure dilutes still fails: on `hard_reset.16k256.lean` a 1 % change of one loud bin on one hop
  (weighted 6.1e-5, green) fails the loud-bin gate (0.010) and the render-product gate (5.2e-3), and a 0.5 % change of every
  bin on one hop (weighted 2.1e-6, loud bins 6.7e-4, both green) fails the render-product gate (5.0e-3).
- **Dumps are validated before any metric.** Every compared array (audio, linear output, trace hops, context records, base
  records, formed hop, delay rows) must be non-empty, the same length as its counterpart and finite on both sides; a
  violation fails the cell whatever its marker (INFO, T-01, expected red). Before this, NaN, +Inf or -Inf in the C output
  passed a cell, INFO cells included (the max-abs reading became "no figure" and `x > tol` is false for NaN), and a NaN in
  `res_gain` or the base aborted the run; an empty or short output already failed through the length flag. Proven on kept dumps: NaN, +Inf and -Inf in the C output, an empty and a one-hop-short output, NaN
  in one `res_gain` entry and NaN in one base bin each fail `hard_reset.16k256.lean` and the T-01 cell
  `never_usable.16k512.lean`. Both drivers run the same hop loop, so no trim is allowed.
- **Whole `--full` run: RESULT PASS**: 48 pass, 11 info (N-01 x2, D-01 x3, N-02, T-01 x5), 3 expected-red (S-02 x3),
  0 fail, 62 scenarios.

## [Unreleased] — 2026-10-28 — C strength setting renamed from preset to level

### Changed
- The C name of the residual-echo strength ladder (`mild` 0 / `balanced` 1 / `aggressive` 2 / `no_floor` 3, default 1)
  is now **level**. Pure rename: the value names, the integers, the per-function out-of-range behaviour
  (`aec_config_from_level` falls back to balanced; `aec_set_level` / `aec_res_set_level` reject with -1) and every output are
  unchanged. A hash of `aec_config_from_level` for 8000/16000/48000 x levels 0..3 (plus out-of-enum values),
  `aec_get_mem_size`, the lookup getters, a 120-hop `aec_process` stream with `aec_set_level` and the standalone RES is
  identical to the pre-change tree. Python is unchanged: it keeps `preset`, so **C level == Python preset**; the function
  map column 1 and the C manual carry the bridge.

  | Old C name | New C name |
  |---|---|
  | `AecPreset` | `AecLevel` |
  | `AEC_PRESET_{MILD,BALANCED,AGGRESSIVE,NO_FLOOR}` | `AEC_LEVEL_{MILD,BALANCED,AGGRESSIVE,NO_FLOOR}` |
  | `aec_config_from_preset` | `aec_config_from_level` |
  | `aec_set_preset` / `aec_res_set_preset` | `aec_set_level` / `aec_res_set_level` |
  | `aec_preset_floor_db` / `aec_preset_dt_floor_db` / `aec_preset_split_floor_enabled` | `aec_level_floor_db` / `aec_level_dt_floor_db` / `aec_level_split_floor_enabled` |
  | `aec_wav` / `bench_rtf` `--preset` | `--level` (`--preset` still accepted) |

- `aec_level_floor_db` / `aec_level_dt_floor_db` return -28 / -20 for `AEC_LEVEL_NO_FLOOR`: an inert placeholder (that
  level disables the floors), not a floor.
- Printed text: the `aec_wav` summary line (stderr) and the `bench_rtf` header line (`# ...`) print `level=` where they
  printed `preset=`; the `aec_run` stderr line likewise. The `parity_aec_e2e` / `aec_run` positional argument and the
  `run_parity_e2e.sh` JSON key `preset` are unchanged.
- `docs/c_user_manual_zh_TW_03_preset.md` is now `docs/c_user_manual_zh_TW_03_level.md` (section 7 anchors follow);
  the test file `c_impl/test/test_runtime_preset.c` and its `make test-runtime-preset` target keep their names because they
  pair by name with `python/tests/test_runtime_preset.py`.

### Kept
- `AEC_PRESET_*` (four enumerators with literal values 0..3) and `typedef AecLevel AecPreset` stay in `aec.h`: the Python
  tests `python/tests/test_preset_arg.py` read those lines. They stay for as long as the Python twins say `preset`. The
  four names are members of `AecLevel` itself (as `AEC_PRESET_*` were members of `AecPreset`), so in C++ they convert to
  `AecLevel` / `AecPreset` and can be passed to `aec_set_level()`; `make test-header-cpp` compiles that.
- The generated comment in `include/aec_balanced_config.h` ("the preset axis") is emitted by
  `python/diag/gen_aec_balanced_config_h.py` and is not edited (`make test-header-fresh`).

### Compatibility
- Link-incompatible for prebuilt objects: the old function names are gone, so recompile and relink callers.
  Struct layouts, enum values and `aec_get_mem_size` are unchanged, so no pool size or layout version moves.
- Python unchanged; stale `preset` wording about the C names remains in Python comments at
  `c_impl/test/pyc_parity/pyc_parity.py:415`, `python/preset_arg.py:1-3,10`, `python/diag/gen_aec_balanced_config_h.py:14`
  and `python/tests/test_preset_arg.py:1,36`.

## [Unreleased] — 2026-10-28 — A/B evidence and simple-mu research tools archived out of the tree

### Removed
- `eval/ab_evidence/` (the 2026-08-06 adaptation-retiming, 2026-08-07 alpha-power, alpha-r and simple-mu sets; 87
  tracked files plus the untracked `simple-mu/listen/` clips), `eval/simple_mu_branch_census.py`,
  `eval/simple_mu_case_probe.py`, `eval/simple_mu_interaction_matrix.py` and their only test,
  `python/tests/test_simple_mu_eval_tools.py` (moved together: the test imports the three tools). They are archived
  outside the repository (see the cleanup record) and stay in git history through `6cf6785`. The archive is not
  runnable as is: the tools resolve imports from their original location.
- `.gitattributes`: it only held the `-whitespace` exceptions for the three archived `source.diff` captures.

### Changed
- `docs/timing_constant_inventory.json` / `.md`: the seven `eval/ab_evidence/<set>/` citations name the commit that
  added the set and the last commit that tracks it; the `.md` is regenerated. No verdict or count changes (80
  candidates, 25 open). `eval/ab_compare.py`, `eval/ab_report.py`, the A/B runners and the 90-case manifest stay.
  No library code or output change; `aec_python_behavior_hash()` is unchanged (9fa7a2de...).

## [Unreleased] — 2026-10-27 — Config factories no longer overflow on an unsupported sample rate; timing-constant inventory cited by symbol

### Fixed
- `aec_config_defaults()` (and so `aec_config_from_preset()`) formed `sr * 64` in `int` for any `sr >= 44100`. A rate of
  33554432 or more overflowed (signed overflow, undefined behaviour): with `-fsanitize=undefined`,
  `src/aec.c:54: signed integer overflow: 33554432 * 64 cannot be represented in type 'int'`. The result depended on
  the optimisation level: at 33554432 `filter_length` was -2147483 at -O1 and 2147483 at -O2, and at `INT_MAX` it was 0
  at -O1 and 4294967 at -O2. The rate was always refused later (`fft_size` is 0 without a grid), so no config was
  accepted, but the factory itself was undefined for an arbitrary `int`.
- `filter_length` is now derived only for a supported rate (`aec_is_valid_sample_rate()`: 8000, 16000, 48000) and is 0
  otherwise, next to the `fft_size = 0` the same factory already gives such a rate. `aec_get_mem_size()` returns 0,
  `aec_init()` returns NULL and `aec_create()` returns nonzero for the result, as for any other rejected config. An
  unsupported rate that used to keep a plausible `filter_length` (44100 gave 2822) now gives 0.
- For 8000, 16000 and 48000 the config is unchanged (`filter_length` 416, 832, 3072; `fft_size` 256, 256, 1024) and
  `aec_python_behavior_hash()` is unchanged (9fa7a2de...). The Python twin was never affected: `AecConfig.__post_init__`
  resolves the grid first and raises `ValueError` for an unsupported rate, and Python integers do not overflow.

- `docs/timing_constant_inventory.json`: one sentence now names the mcab flow like the committed `.md`, so
  `gen_timing_inventory.py --check` (and the three `test_timing_inventory.py` cases that run it) pass again.

### Changed
- `docs/timing_constant_inventory.json` / `.md`: every source citation is now `file:function (`token`)`; the line-number
  citations were stale and are gone. Entries that described code that no longer exists were reworded, and the
  lag-aggregator histogram window entry is closed as already retimed (`da_hist_window_for_rate()` in `delay_aec.c`), so the
  inventory counts 80 candidates, 25 still open (was 26). `python/diag/gen_timing_inventory.py` now rejects a line-number
  citation (an integrity rule with a mutation case in `python/tests/test_timing_inventory.py`).
- `docs/html`: the `aec.c` / `aec.h` line citations of `aec_params.html`, `aec_params_internal.html` and `aec_res.html` follow
  the lines this entry's code change moved; the `filter_length` rows and the usage table name the zero for an unsupported rate.

### Added
- `test_config_validation.c`: `test_config_factories_unsupported_rate()` runs both factories over INT_MIN, -1, 0, 1, 44100,
  33554431, 33554432, 100000000 and INT_MAX and requires `fft_size == 0`, `filter_length == 0` and refusal by all three
  entry points, plus the unchanged factory values at the three supported rates. It fails on the previous `aec.c`
  (10 checks: every rate from 44100 up) and passes with the fix, also under `-fsanitize=undefined`.

## [Unreleased] — 2026-10-24 — Unused C functions removed; streaming FIFO API and its pool storage dropped

### Removed
- The streaming render/capture API: `aec_analyze_render()`, `aec_process_capture()`, `aec_last_buffering_event()`, the
  `AecBufferingEvent` type with `AEC_BUF_NONE`, `AEC_BUF_RENDER_UNDERRUN`, `AEC_BUF_RENDER_OVERRUN` and
  `AEC_BUF_API_SKEW`, and the nine `Aec` fields behind it (render FIFO, its zero reference, capacity, read/write
  indices, call counters, last event) with `AEC_STREAM_FIFO_MS`. No sample, example or pipeline used it; `aec_process()`
  and `aec_process_context()` are the lockstep entry points. The static `next_pow2()` that sized the FIFO went with it.
- `aec_process_context_shared_far()`. `aec_process_context_shared_far_from()` is its superset (it also borrows the
  squared magnitudes) and is what the 4-lane unit calls.
- Forwarding wrappers with no caller outside tests: `delay_aec_accumulate()` (use `delay_aec_accumulate_ex(..., 1)`),
  `shadow_copy_init()` (use `shadow_copy_init_for_grid()`) and `pbfdkf_process()` (use
  `pbfdkf_process_with_far_cmag2()`).
- Macros with no reader: `AEC_BLOCK_MS`, `AEC_LEAKAGE_*_PER_HOP`, `AEC_MAX_ERLE_LF/HF`, `AEC_MIN_ERLE`,
  `AEC_STATIONARITY_*`, `AEC_MATCHED_FILTER_*_LIMIT_FLOAT`, `AEC_NORMAL_RENDER_LIMIT_FLOAT`,
  `AEC_LOW_RENDER_LIMIT_FLOAT`, `AEC_FLOOR_POWER_FLOAT`, `AEC_RESIDUAL_NOISE_GATE_POWER` (`aec_scale.h`),
  `DA_CONSISTENT_EST_THR`, `DA_ACCUMULATED_ERROR_SUBSAMPLE_RATE` (`delay_aec.h`) and `STAT_AVG_INIT_HOPS_DEFAULT`,
  `STAT_INITIAL_PHASE_HOPS_DEFAULT` (`stationarity_estimator.h`).
- `test/stream_sim.c`, `test/test_fifo_spsc.c` and the `test-stream-sim` target. `test_shared_far_spec.c` now drives
  `aec_process_context_shared_far_from()`; its negative control borrows from an owner that processed a different
  reference, and the hop-counter saturation check keeps `frame_count` and `shadow_frame_count`.

### Changed
- Pool size: every `Aec` is smaller by the FIFO storage and the shrunken struct (`aec_get_mem_size()`, balanced,
  default delay mode, SIMD 1): 8 kHz/256 284,352 -> 267,408 B; 16 kHz/256 397,744 -> 364,416 B; 16 kHz/512
  525,440 -> 491,600 B; 48 kHz/1024 1,224,176 -> 1,156,544 B. The difference is the same for every preset and delay
  mode. Audio output is bit-identical (SIMD 1 and SIMD 0: `aec_wav` on 21 clips over 16 kHz/256, 16 kHz/512 and 48 kHz/1024, all four presets at 16 kHz/256).
- `docs/c_user_manual_zh_TW.md`: section 8.4 (separate render/capture entry points) is gone and 8.5-8.8 became 8.4-8.7;
  the pool tables, `aec.html`, `aec_c_api.html`, `STATIC_MEMORY.md`, `c_impl/README.md` and
  `docs/c_python_function_map.md` follow (the map's counts are 409 C functions, 232 mirror, 101 partial, 76 C-only;
  `pbfdkf_process_with_far_cmag2` now carries the `PBFDAF.process` / `complete_update` mapping).
- Public API removals are a breaking change under the versioning rule above: `__version__` in `python/aec.py` goes
  4.0.0rc1 -> **5.0.0rc1**. The behaviour hash ignores `__version__`, so `aec_python_behavior_hash()` is unchanged
  (9fa7a2de...).

## [Unreleased] — 2026-10-24 — Test tiers: lean default, `make test`, `make test-full`

### Changed
- `make test` (previously a no-op, because the `test/` directory exists) now runs the default test tier and
  `make test-full` the complete one. Members run in sequence, failures are listed at the end, and audiolib is
  resolved once. SIMD-independent C targets, `audit-profile`, pytest, the lean pyc suite and the UBSan wrappers run only
  under `SIMD=1`, so the default tier is `make test SIMD=1` plus `make test SIMD=0`. `test-full` adds the full pyc
  matrix, `test-parity-e2e`, the SIMD-independent targets under `SIMD=0`, the UBSan `--use-standard-math` variants,
  `run_fp_flag_policy_test.sh` and `run_ubsan_signal_handling_test.sh`. New target `test-python` runs `python/tests`.
- `make test-pyc-parity` defaults to a 10-scene lean suite (one scene per mechanism a planted regression is known to
  move, including `dtb.16k256.agg` for the aggressive preset and `preecho.16k256.bal.obs_noguard` for the observer
  shift); `PYC_ARGS=--full` (or `pyc_parity_aec.sh --full`) runs all 46 scenes. `--quick` is unchanged.
- `test_res_context_wola` stops 20 hops after the first `filter_converged` rising edge instead of running 2000 hops, and
  `test_aec_reset` warms 200 hops on the two larger grids; both keep their assertions and still fail when the code they
  guard is broken.
- `docs/c_user_manual_zh_TW.md`, `docs/html/aec.html`, `aec_detail.html`, `aec_c_api.html`, `aec_c_api_params.html`,
  `aec_res.html` and `docs/c_python_function_map.md` no longer name the Audio_ALG complete-pipeline wrappers
  (`audio_pipeline_4ch_set_aec_preset()`, `AUDIO_PIPELINE_NOW_US`, the mono pipeline). The residual-floor strength
  of a multi-lane integration is set with `aec_res_set_preset()` on the integrator's own `AecRes`; the mono chain
  keeps using `aec_set_preset()` because its gain is the library's own `res_gain`. No code or output change.

## [Unreleased] — 2026-10-23 — Seam and validation fixes; sources and docs condensed

### Fixed
- Reverting a rejected delay re-lock now bumps the delay generation, so `AecLinearContext.generation` and
  `delay_state` report the change to seam consumers (they stayed at generation 0 / LOCKED). Audio output is unchanged.
- `live_observer_shift_ms` times the hop size is computed in 64-bit; a very large shift now reads delay 0 instead of
  overflowing `int`.
- `aec_realign` with `INT_MIN` samples no longer negates `INT_MIN`.
- `min_gain_split_floor_enabled` is range-checked like the other flags (values other than 0 or 1 are refused).

### Changed
- Internal helpers replace repeated blocks (`aec_preset_row`, `aec_state_bind`, `aec_ree_bind`,
  `aec_epc_gain_change_recovery`, `da_estimator_clear_chain`, `ree_add_reverb`); outputs are bit-identical.
- Documentation: `delay_est_period_s` defaults to 0.0 (analysis every hop) and is a throttle on matched-filter
  analysis; the `enable_shadow = 0` saving is 42,880 B at 16 kHz/256; test counts in the README are current.
  Development-process wording is removed from comments across `c_impl`, `python`, `eval` and `docs`.


## [Unreleased] — 2026-10-23 — Source comments condensed

- `aec.c` / `aec.h`: comments reduced to contracts. Design facts that lived only in comments:
- Delay duty cycle (`delay_est_period_s` > 0) can change which lag the matched filter finds, not only
  the compute, so the default 0.0 analyses every hop. The duty ERLE-watchdog leak is authored as
  0.001 dB/hop on the 10 ms grid (0.1 dB/s); as a literal it would be 0.125 dB/s at the 8 ms grid and
  0.0625 dB/s at the 16 ms grids.
- FIXED reference ring is exactly `fixed_delay_samples + hop_size` (the read needs size >= d + hop;
  equality is safe). The MATCHED ring keeps 4096 samples of headroom, which also bounds the
  `new_delay <= ring - hop` eligibility gate.
- `ne_recent_sustain` stays at the literal 3 and the simple-mu state machine stays hop-authored;
  retiming either to wall clock did not hold across the three grids.
- Bad-relock judgment compares float32 without an epsilon (one ULP exceeds any double slack, checked
  over all finite non-negative values). Rejected candidates are kept until `aec_reset()` in an 8-slot
  list that stops recording when full instead of overwriting.
- External-realign retard exposes echo for up to ceil((|delta| + tap offset) / hop) + 1 hops (worst
  `n_partitions + 1`); for a 256-sample hop and 1024-sample span: delta -300 at tap 0 gives 2 hops,
  -768 at tap 100 gives 4, -900 at tap 60 gives 4, with 1.30x the echo while history refills.
- Live-observer `max_bin` is float32-safe (`bin_hz` is exact on every grid; `ceil()` is stable for
  `live_observer_max_hz` in [1, 24000]); `err_ema_alpha` and `corrob_margin` stay double because they
  drive a double-precision per-hop recursion.
- Streaming FIFO capacity is a power of two (32 hops at every supported rate) so unsigned cursor
  indexing is continuous across 2^32. The consumer releases a slot only after `aec_process()` returns,
  which can report one extra drop-new under sustained skew; the design is TSan-clean.
- Stage timing costs seven clock reads per hop per instance; the five buckets do not sum to the call
  (power EMAs and convergence run after `res_us`); the 32-bit stamp wraps at about 71.6 minutes.
- `min_gain_floor_dt_db` = -20 dB is an operating point that recovers residual-echo suppression lost
  when the near-recent latch false-arms in far-end single talk, at a small double-talk cost.
- Other C sources (`src/*.c`, `include/*.h` except `aec.c` / `aec.h` / `aec_balanced_config.h`,
  `example/`): comments reduced to contracts. Design facts that lived only in comments:
- SC_HYS_STREAK_MIN is a hop count on every grid; retiming it to 100 ms (12 hops at 16 kHz/128)
  broke a near-end single-talk case (the rescue streak stopped completing and the output stayed ~5
  dB under the capture); 11 fails too, 10 passes.
- Counter saturation: raw counter values are observable, so they stop at the integer limit, not at
  their gate threshold, where a threshold cap would still satisfy the comparison
  (`hold_counter_time_domain` at INT_MIN, filter-quality counters and `call_counter` at INT_MAX /
  LONG_MAX, the filter-analyzer consistency counter at LONG_MAX). The fullband-ERLE hold counter
  also stops at INT_MIN rather than 0 because its `== 0` check is an edge trigger that must fire
  once per re-arm; flooring at 0 would re-fire it every hop. Counters read only by a threshold
  comparison (`blocks_since_reset`, `stability_counter`) stop one past the threshold.
- Stationarity flags walk the history row-major over 128-bin accumulator blocks (STAT_ACC_BLOCK):
  128 keeps the frame near 1 KiB and is below the smallest shipped bin count so the blocked walk
  splits on every grid; about 6% slower than a 256-bin accumulator at 129 bins, about 4x faster than
  the column-major walk.
- Live observer corroboration compares the observer's low-frequency residual with the main filter's
  residual on the same hop (EMA pair, margin observer_corrob_margin); comparing against the
  observer's own past state cannot tell correct extra cancellation from near-end contamination,
  since both lower its own residual. Evidence is capped at min(|observer.echo_spec - sel_echo|^2,
  |sel_esw|^2) and limited to observer_max_bin; the EMA recursion and comparison run in double.
- DA_MAX_FILTER_LAG_FOR(n) = n * intra_shift + filter_size deliberately keeps one filter of headroom
  over the reachable lag (the same sizing formula as the reference matched filter); n = 5 gives 2433
  highest-peak and 152 pre-echo histogram bins. The extra tail is behaviour-neutral: bins above the
  reachable lag are never incremented and ties resolve to the first maximum.
- The reported delay is always the aggregator's dominant highest-peak candidate (reference design
  reports the earliest onset); detect_pre_echo has no effect here, so the pre-echo buffers keep
  their initialised values. Reporting an earliest-onset candidate would attach the dominant
  histogram's confidence to a delay it does not describe (a 100 ms-early onset once reached a 52 ms
  PBFDKF with confidence 1.0).
- Delay chain is float32 end to end (decimator biquads, matched-filter dot products and NLMS update,
  anchors); sums accumulate directly in float32 (block-wise on NEON).
- Highest-peak histogram argmax is maintained incrementally: while a tracked candidate c (first
  index of the maximum M) is valid, a decrement on another bin leaves (c, M) unchanged, and an
  insert at B only replaces c when it exceeds M (or ties it at a lower index); when the candidate
  itself is decremented the new value must be compared against the old M, otherwise an untracked tie
  is missed (for example H=[3,5,5,2,0], candidate 1, evict bin 1 and insert at bin 3 gives argmax
  2), so that case rescans. A decrement and increment on the same bin are intercepted by the caller
  as a net-zero update.
- example/wav_io.h resolves the shared audiolib wav_io.h through explicit relative __has_include
  paths (sibling checkout, or two levels deeper in an Audio_ALG submodule). #include_next was
  rejected because for a same-directory include it restarts the search from the top and re-finds the
  shim; the shim's guard is therefore not WAV_IO_H.
- Python library (`python/aec.py`, `python/modules/**`): comments and docstrings reduced to contracts;
  behaviour hash unchanged. Design facts that lived only in comments:
- Per-block EMA constants in the fullband and subband ERLE estimators fire once per 6-hop accumulation
  window; the factor 6 cancels, so they are rescaled with a single hop (passing 6 x hop inflated the
  exponent 6x). The 0.0004 max/min forget is an additive per-block step and is rescaled linearly.
- `PathChangeRegimeHandler` is the defence for wildly nonstationary echo paths: without it the main
  filter diverged to anti-correlated output on a p99.2 ERL-decile-std case (ERLE_main -27 dB); its
  boost_q / pause_main / reverse_copy actions drive W toward 0 (pass-through) instead. `HYS_STREAK_MIN`
  is 10 hops on every grid (see the C bullet above). The regime classifier thresholds 9.43 / 21.04 dB
  are the p90 / p99 of ERL-decile std over the 800-case corpus (660 cases with enough far-active
  deciles; the reference case measured 23.39 dB, rank 655/660); the classifier is analysis-only.
- `StationarityEstimator`: a constant ~0.009 RMS far-end hum made two near-end single-talk cases lose
  9-18% of near-end RMS, hence R^2 is zeroed on stationary bands once the filter has converged.
- `ReverbFrequencyResponse` keeps one-bin windows and the in-place ascending neighbour-max loop: widening
  the conservative-max and neighbour-max windows to +-125 Hz at fft 512 inflated the HF tail
  (reverb/direct ratios up to 1e6, "painted black" HF during double-talk) and was reverted.
- `H_error` constants stay in AEC3's own scale (K * E is scale-invariant; rescaling by 1/PSD_SCALE
  collapsed learning: FS echo 3.804 -> 2.311, ERLE ~10 -> ~2 dB). The cold-start deadlock breaker
  (`h_error_refresh_erl_floor`) exists because 67% of far-end single-talk cases never converged
  (mean ERLE 0.14 dB) when W ~ 0 gave no H_error refresh.
- Preset floors (amplitude dB, far-active / DT): mild -20/-20, balanced -28/-20, aggressive -38/-30.
  800-case at balanced: FS echo 3.549, DT echo 4.155, DT deg 2.183, NE deg 4.047. DT floor -20 vs -16
  (832-file paired): FS echo MOS +0.053/+0.064, ERLE +0.46/+0.53 dB, DT deg -0.027/-0.030. Aggressive's
  -30 DT floor (47-case dev pool): DT_static echo +0.17/+0.14, deg -0.16/-0.14, near-end 200 ms guard
  42/48 -> 39/48 (grid A) and 44/48 -> 43/48 (grid B).
- Other defaults and their measured effect (800-case): round-robin constraint FS echo +0.055/+0.073,
  DT echo +0.043/+0.055 at -0.031 DT deg; soft near-end blend FS echo +0.042, DT deg -0.040; second-order
  nonlinear residual FS echo +0.079/+0.106, DT deg -0.027/-0.031 (cap 300x leaves output unchanged and
  bounds the worst r2_nl 2.76e18 -> 6.52e15); ERLE render PSD scale revives the reverb model (linear-filter
  quality 0 -> 32%, tail 0 -> 73%, far-offset residual -18 -> -30 dB); windowed ERLE Y2 FS echo +0.08, DT deg
  -0.05; delay_acquire_protect_converged changed 26 cases (net echo +2.85); warm tap-transfer fires on ~117 of
  600 FS+DT cases (DT deg +0.089 on that subset).
- Live-observer injection ceiling `live_observer_inject_dt_energy_max` = 0.5: the corroboration margin cannot
  separate false-positive injections from legitimate ones (EMA ratio medians 0.82 vs 0.84) and a consecutive
  streak gate failed; 0.3 cost fid0 grid B ERLE +7.12 -> +5.70 dB, 0.5 costs +7.12 -> +6.91 dB and removes the
  DT_movement deg loss on the worst case (-0.117 -> +0.0008).
- Delay chain: the 48 kHz path first runs an anti-alias sidechain (7th-order elliptic, passband 6800 Hz /
  1 dB, stopband 8000 Hz / 60 dB, decimate by 3), because without it the Python and C estimators locked
  different delays at 48 kHz. The clock-drift stability window and the lag-aggregator histogram are
  counted in inner 64-sample blocks (30 s window, `sample_rate / 64` estimates), not in hops.
- Warm delay realign: keeping `X_buf` on an advance beat clearing it (line removal -16.8 vs -23.9 dB,
  steady 11.5 vs 12.2); a retard clears it. Soft Path-B realign shifts the IR by the applied delta because
  without it raw output reached 2.2-2.4x full scale on a real file. A second Path-B shift inside a pending
  bad-relock judgment overwrites that judgment (resolving early or refusing the shift were rejected).

## [Unreleased] — 2026-10-23 — Smaller pool, fewer copies in the weight constraint and the delay chunker

- `aec_get_mem_size()` is 1,024 / 2,048 / 4,096 B smaller on the 256 / 512 / 1024 grids (the FFT handle
  stages its preserving transforms in one block); the NLMS/Kalman time-domain constraint hands the weight
  row to `fft_inverse_scratch()` (the forward transform that follows rewrites it) and the delay estimator's
  64-sample input chunker appends in `memcpy` runs, keeping the one-sample path for an input that overlaps
  a pending array. Output and all state are bit-identical; the C user manual, `STATIC_MEMORY.md` and the
  HTML pages carry the re-measured byte counts, and `test_delay_reset` gains split-invariance and overlap
  checks.

## [Unreleased] — 2026-10-23 — Forwarding wrappers removed; 32-bit ARM test builds clean

- The one-line wrappers over the shared pairwise-sum and magnitude kernels are gone and their callers
  use the kernels directly (output bit-identical): `cabs_np()`, `f32_pairwise_sum()`,
  `fb_erle_pairwise_sum()`, `fa_f32_pairwise_sum()` and `aec_post_pairwise_sum_f32()` are no longer
  exported. `ck_field_size_reps()` moved to `audiolib/common/utility/mem_align.h`. With
  `-mfpu=neon` / `-mfpu=neon-vfpv4` and gcc 12, `make selftest` builds under `WERROR=1` and runs under
  qemu-arm, `test_shared_far_spec.c` no longer overflows a 32-bit `long` constant, and the printed
  standalone build recipes of the tests and `docs_smoke.sh` carry the `-I` they need.

## [Unreleased] — 2026-10-23 — Unused internals removed; the AEC pool is smaller

### Removed
- `PBFDAF.scr_e2` (one `n_freqs` float array per filter, read by nothing) and its carve. With the
  `is_static` fields below, `aec_get_mem_size()` of the default balanced configuration drops by
  1,120 B at 16 kHz/256, 2,144 B at 16 kHz/512 and 4,192 B at 48 kHz/1024 (observer off; 1,648 /
  3,184 / 6,256 B with it on). Callers that keep a descriptor must query the size again.
- The heap constructors `pbfdaf_init()` / `pbfdkf_init()` and the heap branch of
  `pbfdaf_free()` / `pbfdkf_free()` (now only dropping the borrowed shared-FFT reference), with the
  `PBFDAF`/`PBFDKF` `is_static` fields. `test_config_validation` keeps proving the `sample_rate`,
  `hop` and NULL guards through `pbfdaf_init_static()` / `pbfdkf_init_static()`, and now also checks
  that every refusal leaves the instance and the pool untouched (404 -> 405 checks).
- The `FilterPlateauDetector` struct and `filter_plateau_*()` (no production caller; the Python
  class is unchanged) and the `test-counter-saturation` section that only exercised them
  (12,081 -> 6,344 checks; every other counter check is kept).
- `test/support/reverb_decay_estimator.{c,h}` (not built, not called) and
  `python/diag/gen_reverb_decay_estimator_golden.py`.

## [Unreleased] — 2026-10-23 — Frozen-digest regression checks are no longer shipped

### Removed
- `make test-output-hash` and its sources (`test/test_output_hash.c`, `test/golden/output_hash.txt`),
  `test/golden/sg_config.txt`, and the delay-chain replay pair `test/parity_delay.c` /
  `test/gen_delay_c_golden.c` with its generator `python/diag/gen_delay_golden.py`. They compared the
  C output with digests of earlier C output; the context-level invariance of the linear outputs stays
  in `make test-process-context`.
- `make test-aec-res` no longer takes a golden argument: the configuration digest check and the
  comparisons with transcriptions of the former pipeline RES stage and comfort-noise loops are gone
  (the lane-equal configuration, API contract and stage-detail checks remain; 225 checks).
- `make test-spectral-reuse` lost its before-API build branch and the `ARM_REUSE_DUMP` dump.
- `pyc_parity_aec.sh --diff` / `pyc_parity.py --diff` (comparison of two saved result files).
- `test/support/sha256.h`, which only the removed digest checks used.

## [Unreleased] — 2026-10-22 — Matched filter runs on NEON for 32-bit ARM with fused FMA

- `delay_aec.c`: when `__arm__ && __ARM_NEON && __ARM_FEATURE_FMA` (for example
  `-mfpu=neon-vfpv4`) the dot, NLMS update and fused update-plus-next-dot kernels use
  NEON with the same four-accumulator tree as AArch64 (`vaddvq_f32` is replaced by two
  `vpadd` steps giving `(l0+l1)+(l2+l3)`). Results are bit-identical to AArch64 NEON for
  finite, non-subnormal inputs; ARMv7 NEON flushes subnormals to zero and returns the
  default NaN.
- Cores without fused FMA (`-mfpu=neon`, `neon-fp16`, `vfpv4`) keep the scalar path;
  `-DDA_DISABLE_ARM32_NEON` forces it.
- New target `make test-delay-matched-neon` (bit-exact against a portable `fmaf` model).
- AArch64 builds are unchanged (identical object code).

## [Unreleased] — 2026-10-22 — Hop and call counters saturate

### Fixed
- `frame_count`, `shadow_frame_count`, `render_call_count` and `capture_call_count`
  hold at `LONG_MAX` instead of overflowing signed long on ARM32, like the far-FFT
  counters. Public types and every count below the limit are unchanged.
- The shadow hop count is handed to the shadow double-talk update and the shadow-copy
  decision as `int`. A count past `INT_MAX` (reachable only where `long` is 64 bits)
  is now passed as `INT_MAX` rather than narrowed modulo 2^32, which made it negative
  and re-closed the warm-up gate. Results are identical for every count up to `INT_MAX`.
- `test_shared_far_spec` covers all four counters at the limit and their reset, and
  compares an instance seeded past `INT_MAX` with one seeded inside int range.

### Documented
- ARM32 builds should pass `EXTRA_CFLAGS=-mfpu=neon-vfpv4` (`-mfpu=neon-fp-armv8`
  on AArch32 ARMv8), not plain `-mfpu=neon`: without hardware FMA each explicit
  `fmaf()` becomes a libm call. ARMv7 Advanced SIMD flushes subnormals and returns
  the default NaN, so `SIMD=1` and `SIMD=0` can differ on ARM32 for such values (NE10
  NEON FFT only).

## [Unreleased] — 2026-10-22 — Render analyzer uses the shared max-abs kernel

### Changed
- `rsa_max_abs()` / `rsa_max_abs_scan()` in `render_signal_analyzer.c` now call
  `sk_max_abs_f32()` / `sk_max_abs_f32_scalar()`. The two implementations agreed on
  every NaN-free input; the only difference is a NaN at `render_block[0]`, which no
  longer blocks the strong-peak latch (NaN never raises the maximum, at any index).
  A non-finite render sample is outside the input contract and already corrupts the
  filter state in the same hop.

## [Unreleased] — 2026-10-22 — ARM32 heap alignment

### Fixed
- `aec_create()` requests a 16-byte aligned arena explicitly. ILP32 `malloc()`
  may guarantee only 8-byte alignment, while the shared static carve requires
  16 bytes. Allocation ownership and `free()` remain unchanged.
- GCC builds keep the scalar delay dot loop within a signed-safe grouped bound
  and reject invalid render-detector lengths before reading its local buffer.
  Valid hops and floating-point operation order are unchanged.
- Far-FFT instrumentation increments and aggregate sums saturate at `LONG_MAX`
  instead of overflowing signed long on ARM32. The public type and normal
  counts are unchanged; `test_shared_far_spec` covers the limit and reset.
- The static-AEC test retains its input audio through its leak-check loop.
- Caller-pool tests request 16-byte alignment explicitly. The heap allocator
  regression forces an 8-byte-only malloc base, checks allocation rejection
  before state writes, and checks balanced release across the three product grids.
- The spectrum-reuse regression also checks normal heap arena alignment. Pool
  sizes and structure layouts must be queried with the consuming target ABI.

## [Unreleased] — 2026-10-21 — Shared AEC header and current documentation

### Changed
- AEC C consumers include `utility/simd_aec.h` directly; the empty
  `aec_simd_kernels.h` compatibility forwarder is retired. Shared kernels
  remain owned by Audio_ALG's function map rather than duplicated in AEC's map.
- Fixed the standalone UBSan self-test and counter-saturation probes' missing common-root include path.
- Trimmed redundant C comments while retaining numeric order, NaN, aliasing,
  counter and buffer-lifetime contracts. C tokens and expanded translation
  units are unchanged apart from the explicit include-path redirect.
- Current timing-inventory consumers cite verified function and symbol anchors
  after the stage extraction. All 80 historical provenance fields, verdicts
  and effective-value fields are unchanged. CLI help now documents preset 3.
- Persisted AIAEC preset contracts and NR benchmark strength records keep
  their string names. Benchmark comparison also accepts stored numeric values.

- Removed the unused private `cmag2_c` helper after confirming it has no callers; retained test-only scalar wrappers.


## [Unreleased] — 2026-10-21 — Ordered C/Python AEC stages

### Changed
- Split the C process core and pool constructor into 34 private helpers;
  split Python construction, per-hop processing and residual orchestration
  into 57 private helpers. Public APIs, arithmetic and stage order, filter
  ownership, queue/reset behavior and the pool byte layout are unchanged.
- The Python engine CLI accepts preset values `0/1/2/3` and the corresponding
  `mild/balanced/aggressive/no_floor` names through the shared parser.
  Persisted dataset presets remain strings; frozen materialization stays
  `balanced` with the same signal grid and feature gates.
- The combined Python signal identity is now `9fa7a2de...245e5` (previously
  `419f473e...ff8fa`). The AST canonicalizer and `canon-ast-1` schema are
  unchanged. No corpus migration was added: there is no local ledger at the
  previous identity needing rescue, and retired hashes remain refused.
  The separate released-checkpoint inference pointer advances to this build;
  its earlier, unmeasured history remains explicitly documented.

### Validation
- All 24 frozen C output digests and 6 context-invariance checks pass in both SIMD modes. The new memory-layout
  test covers 72 allocation combinations, including exact pool usage,
  undersized rejection and tail canaries. Streaming stack bounds and the
  independent stack-canary peak fall by 48 bytes; construction grows by
  48 bytes while the overall maximum falls by 48 bytes.
- Python renders 8,636 real-audio hops across nine 16/48-kHz frozen,
  full-suppression/CNG and observer profiles with byte-identical materialized
  error, formed output, echo, context and state traces. Eight controlled
  public-process recovery scenarios reach all 57 helpers and also retain
  exact output/state/alias identity. The same comparator rejects a one-bit
  mutation at the materialization getter. Existing Python tests, 22 new
  CLI/diagnostic contract tests and the C/Python map guards pass.
- Four paired Python timing runs over nine real profiles have median
  after/before ratios 0.9704–0.9942; paired C full-path ratios are
  0.9746–1.0177 across the three supported grids. No material slowdown is
  observed; small timing differences are not promised speedups.
- Full live AIAEC validation: 727 passed, 2 skipped, 3020 warnings in 730.42s.
  Detailed evidence is under
  `SE_verify_20261019/scratch/codex_f_python_20261021/` and
  `SE_verify_20261019/scratch/codex_f_c_20261021/`.

## [Unreleased] — 2026-10-21 — Shared periodic sqrt-Hann window

### Changed
- The adaptive filter builds its periodic sqrt-Hann through audiolib's
  `audio_window.h`; the causal constraint window and filter state are unchanged.
  The original and shared float32 tables are bit-identical at every supported
  grid. The `Audio_ALG/lib/aec` working-tree copy is mirrored byte-for-byte.

## [Unreleased] — 2026-10-20 — NO_FLOOR preset (`AEC_PRESET_NO_FLOOR` / preset `3`): C-side ABI landed

Additive only: `MILD`/`BALANCED`/`AGGRESSIVE` are byte-for-byte unchanged. Verified three ways: (1)
`test/golden/sg_config.txt` and `test/golden/output_hash.txt` unmodified on disk and both still pass
against the checked-in digests; (2) 19 `test-*` targets re-run under both `SIMD=0` and `SIMD=1`, all
green in canonical `SE/AEC` (`test-cli-delay-flags`/`test-static-aec`/`test-parity-e2e` were run once,
default build only -- not `SIMD`-parameterized in a way distinct from it); (3) a direct `pyc_aec_run` E2E check --
mild/balanced/aggressive x the three product grids (16k/256, 16k/512, 48k/1024), `.out.f32`/`.raw.f32`
sha256 before vs after this change, byte-identical on all 18 files, in both this repo and
`Audio_ALG/lib/aec`. `AecConfig` grows by one `int` field (`sizeof` 220 B -> 224 B), appended at the
struct's end per the existing `delay_mode`/`live_observer_*` ABI-stability convention, so every field
before it keeps its byte offset.


The `Audio_ALG/lib/aec` mirror has the same source but lacks the gitignored local
`wav/aec_challenge_blind/` fixture: 17/19 targets pass there; `test-static-aec` and
`test-parity-e2e` stop at the pinned-input checks (`TEST_STATIC_MIC not found` /
`pinned input check failed`). The canonical checkout retains that fixture and
passes both targets. The `NO_FLOOR` replay likewise passed the canonical 16 kHz
and 48 kHz goldens inside `TOL_E2E` / `TOL_E2E_48K`; the mirror was verified
byte-identical but cannot replay that absent fixture.

### Changed

- Extracted eight AEC inline SIMD blocks into `audiolib/common/utility/simd_aec.h`
  with explicit scalar twins: H-error leakage, ERLE accumulation/release,
  render peak counters, stationarity hangover/smoothing/thresholds, and
  saturation counts. Local wrappers and per-bin arithmetic order are unchanged.
  The AEC selftest covers lane tails, 129/257/513 bins, state thresholds,
  non-finite predicates and write canaries. Both SIMD modes retain all 30 output
  digests; scalar and NEON mutation checks independently reject broken paths.
- Removed 159 stale historical-probe links from the C/Python map after the
  already-retired probe files were confirmed absent; current correspondence
  rows and live test links remain intact.

### Added

- **`AEC_PRESET_NO_FLOOR = 3`** (`c_impl/include/aec.h`), mirroring Python's `AecPreset.NO_FLOOR`
  (`"no_floor"`), which already supports this preset at construction time only (`config.py` /
  `orchestrator.py`, landed in an earlier merge, `02df7cd`). `AecConfig` gains `int
  min_gain_split_floor_enabled` (default `1`, filled by both `aec_config_defaults()` — so a caller
  that never goes through a preset at all still gets the floor enabled, instead of the `memset`
  default of `0` — and `aec_config_from_preset()`). `aec_build_sg_config()` (the single construction
  function shared by `Aec`'s own path and `AecRes`'s, via `aec_res.c`'s `resolve()`) now reads
  `cfg->min_gain_split_floor_enabled` instead of the previous hardcoded `scfg.split_floor_enabled =
  1;` — one edit site wires construction-time `NO_FLOOR` into both call paths.
- **`suppression_gain_set_split_floor_enabled(SuppressionGain*, int)`** (`suppression_gain.h`/`.c`):
  steps immediately, no ramp (a boolean has no continuous path to ramp along — same discipline as
  the existing `suppression_gain_set_split_floor_dt_db()`). Gates all three floor variants in
  `get_min_gain` together (far-active, far-silent/pure-NE, DT-lifted). Does **not** reset
  `_far_active_latched`: a latch that could not arm while disabled resumes from wherever it was, not
  from scratch — this is the one behavioral nuance a runtime `NO_FLOOR` round trip does not erase.
- **`aec_preset_split_floor_enabled(AecPreset, int*)`**: public lookup for the gate alone, mirroring
  `aec_preset_floor_db()` / `aec_preset_dt_floor_db()`'s contract, so a caller that owns a
  `SuppressionGain` outside any `Aec` (the 4-channel core's shared post-stage suppressor) can read
  the gate without the two dB lookups. (The 4-channel pipeline's own consumption of this is a
  separate, later change — not made here.)
- **`aec_set_preset()` / `aec_res_set_preset()` now accept `AEC_PRESET_NO_FLOOR` at runtime, with no
  accept/reject guard.** C never had a rejection guard for this in the first place (it never had a
  4th enum value before this change); switching is simply allowed. Going INTO `NO_FLOOR` steps the
  gate off on the same call; going AWAY FROM `NO_FLOOR` steps the gate back on **and** writes the
  target preset's far-active/dt dB values through the existing (ramped/stepped) setters in the same
  call, so a round trip never lands on a half-applied mix of "gate on, dB still at `no_floor`'s inert
  value" or vice versa. `aec_res_set_preset()` also mirrors the gate into `r->sg_cfg.split_floor_
  enabled`, matching the existing `split_floor_far_active`/`split_floor_dt` mirrors — this is what
  `aec_res_reset()`'s `rebuild()` re-inits `suppression_gain_init()` from, so a missing mirror write
  would have passed every other check and only failed on reset (covered by a dedicated test, see
  below).
- CLI: `example/aec_wav.c`'s `--preset` accepts `3` / `no_floor`; usage string and the preset-name
  trace line updated for the 4th case. The remaining three CLI surfaces this entry originally left
  out are now also wired: `test/bench_rtf.c`'s `--preset` (`parse_preset()`/`preset_name()`/usage/
  error text), `test/test_rate_structural.c`'s hardcoded preset-sweep array (`{MILD, BALANCED,
  AGGRESSIVE}` -> adds `NO_FLOOR`; the "presets don't affect `aec_get_mem_size()`" invariant this
  array drives holds for the 4th preset too — verified directly, not assumed), and
  `test/parity_aec_e2e.c`'s `argv[2]` preset parse (previously fail-loud with "no no_floor case
  wired up" — now accepts `3`/`no_floor` and replays correctly: a `no_floor`-preset golden from
  `gen_aec_e2e_golden.py` passes both grids tested, 16 kHz and 48 kHz, with linear/output maxima in
  the same range as `balanced`'s, well inside the existing frozen `TOL_E2E`/`TOL_E2E_48K`).
- Tests: `c_impl/test/test_runtime_preset.c` gained `run_no_floor_switch()` (INTO/AWAY-FROM, each
  landing field-for-field on a freshly constructed reference instance) and `run_no_floor_round_trip()`
  (BALANCED -> NO_FLOOR -> AGGRESSIVE, gate and both dB scalars checked at each leg); `test_aec_res.c`
  gained the `AecRes`-side NO_FLOOR block next to its existing preset/reset coverage, including the
  reset-survives-the-gate assertion described above. `test/pyc_parity/pyc_parity.py` gained two new
  scenario rows (`dtb.16k256.nofloor`, `dtb.48k1024.nofloor`); both pass Layer 1 and Layer 2 cleanly
  within the existing frozen tolerances (margins 0.0041–0.02x of tolerance — no tolerance constant
  was touched). `test_rate_structural.c`'s sizing sweep re-run clean (386/386) with the 4th preset
  added; `bench_rtf --preset no_floor` smoke-tested directly (full/linear/post stages all report).

### Known gap — not closed by this entry

Python's `AEC.set_preset()` still **rejects** a runtime retarget to or from `NO_FLOOR` (the
`orchestrator.py` guard from `265f718`/`b60a1d1` is untouched) — only `AecConfig.from_preset(AecPreset.
NO_FLOOR)` (construction-time) is supported on the Python side. C now allows what Python still
refuses; porting Python to match (removing its guard, adding the equivalent boolean-step setter) is a
separate, not-yet-landed change. Construction-time `NO_FLOOR` is already symmetric between both
languages (confirmed directly: the new `pyc_parity` `nofloor` scenarios above compare C's and Python's
`NO_FLOOR` construction against each other and pass).

Doc figures regenerated for the ABI growth: `STATIC_MEMORY.md`'s two balanced-preset pool-size
figures (16 kHz/256: 399,872 B -> 399,888 B; 16 kHz/512: 529,616 B -> 529,632 B — +16 B, not +4 B,
because the pool carve rounds up to a 16-byte boundary) and every absolute byte figure in
`docs/c_user_manual_zh_TW.md`'s memory-pool tables (§ "記憶池大小"), re-measured via `make
test-delay-num-filters` and `aec_wav --print-mem-size`, not hand-patched — every per-mode/per-n/
per-grid *delta* is unchanged, only the absolute totals moved. `docs/c_user_manual_zh_TW.md` §7/§7.1
and `README.md`'s preset table/retargeting section gained the 4th preset and the C/Python asymmetry
note above.

## [Unreleased] — 2026-10-20 — NO_FLOOR Python-side runtime setter + S-05 fix, landed as one combined `aec_behavior_hash` migration

Closes the "known gap" left open by the C-side NO_FLOOR entry directly above: `AEC.set_preset()` now
accepts `AecPreset.NO_FLOOR` at runtime, in both directions, matching C's `aec_set_preset()` exactly.
Landed together with an independent, unrelated fix (S-05) because both touch the hash-covered surface
(`aec.py` + `modules/**`) and this campaign's migration table is a single hop, not a chain — two
separate entries would not compose.

### Added — NO_FLOOR runtime setter (Python)

- **`SuppressionGain.set_split_floor_enabled(bool)`** (`modules/residual/suppression_gain.py`): steps
  immediately, no ramp — the same deliberate asymmetry as the existing `set_split_floor_dt_db()`, and
  the direct mirror of the C-side entry above's `suppression_gain_set_split_floor_enabled()`. No
  validation needed; any truthy/falsy value coerces, matching `set_initial_state()`'s own contract.
- **`AEC.set_preset()` now accepts `AecPreset.NO_FLOOR`, with no accept/reject guard.** Both the
  identity-check guard (`265f718`/`1b9e5b2`) and, in this repo only, a redundant duplicate from a
  prior merge (`02df7cd`) that also carried the later symmetric property-check guard (`b60a1d1`) are
  removed. Going INTO `NO_FLOOR` steps the gate off on the same call; going AWAY FROM `NO_FLOOR` steps
  the gate back on **and** writes the target preset's far-active/dt dB values through the existing
  setters in the same call — a round trip never lands on a half-applied mix of the two presets'
  behavior. `Audio_ALG/lib/aec/python` never carried the duplicate-guard merge artifact (it only had
  the single `b60a1d1`-style guard), so its edit removes one guard block where this repo's removes two;
  the two files converge to the same `set_preset()` body as a result — this is expected convergence
  from removing the only pre-existing textual difference between them, not a forced rewrite.
- `_far_active_latched` (`suppression_gain.py`) is deliberately **not** reset or force-armed on a
  gate transition: the same "lazy catch-up, cheapest and least disruptive" choice the C-side entry
  above made, kept here for C/Python parity rather than independently re-decided.
- Tests (`tests/test_runtime_preset.py`, identical in both repos): `test_set_preset_rejects_no_floor`
  and `test_set_preset_rejects_retargeting_away_from_no_floor` are rewritten to
  `test_set_preset_allows_no_floor` / `test_set_preset_allows_retargeting_away_from_no_floor` (same
  scenarios, opposite expected outcome); added `test_set_preset_no_floor_round_trip` (BALANCED ->
  NO_FLOOR -> AGGRESSIVE, gate and both dB scalars checked against a fresh instance of the target
  preset at each leg) and `test_set_preset_no_floor_gate_steps_without_ramp` (a large `ramp_ms` must
  not affect the gate). Mutation-tested: temporarily restoring the old rejection guard in
  `set_preset()` turns all four of these tests RED; reverting the mutation turns them GREEN again
  (full suite: 42/42 passed in both repos, both before-mutation and after-revert).

### Fixed — S-05 (`AecResContext.filter_converged` one hop late on the rising edge)

- `AEC.process()` built `AecResContext` (when `return_res_context=True`) **before**
  `self._convergence.update_convergence()` ran for that hop, so `context.filter_converged` always
  reported the *previous* hop's convergence state. Every hop except one was unaffected (convergence
  is monotonic once reached, so the stale read matched the fresh one almost everywhere) — the one hop
  that differed was the exact rising edge, where convergence transitions `False -> True`: the context
  returned for that hop said "not converged" one hop later than it should have. C's twin,
  `aec_get_res_context()`, is a separate pull the caller makes only *after* `aec_process()` returns for
  the hop, so it was never exposed to this — it is always implicitly post-update. The fix patches
  `_res_context.filter_converged` right after `update_convergence()` runs (same hop, nothing else
  touched), bringing Python's single-call return in line with that same post-update contract. No other
  field, no filter/gain/mu-scale behavior, and no other consumer of `self._convergence.converged`
  changes — this is a pure read-timing correction of one diagnostic output field.
- Confirmed via the C-vs-Python `pyc_parity` harness (`test/pyc_parity_aec.sh --only
  dtb.16k256.bal.ctx,dtb.48k1024.bal.ctx,speechdt.16k256.bal.ctx_nores`): all three scenarios carried
  the `EXPECTED_RED(S-05)` marker before this fix (`ctx_conv_mismatch_hops` > 0, shape = late rising
  edge only) and now show `ctx_conv_mismatch_hops=0` and a clean `PASS`. The marker and its
  `EXPECTED_RED_INFO` entry are removed from `pyc_parity.py` (identical in both repos); leaving it in
  place would have made the harness itself fail once fixed (`FAIL UNEXPECTED_GREEN`/"S-05 shape lost",
  by the harness's own self-check). A `--quick` run of the other 10 scenarios shows no other
  regression (`RESULT: PASS`).
- Test (`tests/test_res_context_wola.py`, identical in both repos):
  `test_res_context_filter_converged_tracks_this_hop_not_last` drives a synthetic near-ideal echo
  scene (no near-end, no noise) for 2000 hops at 16 kHz/512, and on every hop asserts
  `context.filter_converged == aec.get_stats().filter_converged` (the latter is read from
  `self._filter_converged`, by construction always post-update once `process()` has returned) — the
  scene is checked to actually reach the convergence rule's rising edge, not just run long enough that
  it probably does. Mutation-tested: disabling the patch reproduces exactly the documented failure
  mode (`AecResContext.filter_converged (False) disagrees with ... (True)`, on the rising-edge hop);
  restoring it passes again (full suite: 7/7 passed in both repos, both before-mutation and after-revert).
- `docs/c_python_function_map.md` (identical in both repos): the `S-05` and `S-20` gap tags are
  removed from every row and from the appendix definition list (the pattern this doc already uses for
  a closed gap — compare `S-01`, which does not appear anywhere in the current table either), the two
  new C functions from the entry above (`aec_preset_split_floor_enabled`,
  `suppression_gain_set_split_floor_enabled`) and this entry's new Python function
  (`SuppressionGain.set_split_floor_enabled`) get rows, three stale "NO_FLOOR 僅 Python 有" / "C 的列舉
  沒有 NO_FLOOR" notes left over from the C-side entry above are corrected, and the `§0.4` stats block
  is regenerated by the document's own guard test (`c_src_total` 427->429, `c_src_mirror` 265->268,
  `c_src_partial` 75->74, `c_total` 461->463, `c_mapped` 341->343, `py_total` 512->513, `py_mapped`
  303->304), not hand-computed.

### `aec_behavior_hash` migration (combined, single hop)

Both changes above touch `modules/**`, which `AIAEC/dataset_gen/aec_behavior_hash.py` hashes; neither
is callable from any dataset-generation code path (`set_preset`/`set_split_floor_enabled` are never
called from `AIAEC/dataset_gen/*.py`, confirmed by grep; the `AecResContext` instance `LinearAecProcessor`
discards is never read back). Registered as **one** combined migration, per
`linear_aec.py`'s own "single hop by construction" rule — a second entry stacked on top of a first
would not compose.

- **Source** `aec_python_behavior_hash()` on `Audio_ALG/lib/aec/python`: `428aa9b796c2491a9cb7baaff82a5c1ba3fc6da4f8e04a38d0676458bf1a562d`
  (re-measured immediately before this change; matches the value already on record — nothing else had
  landed in `modules/**` or `aec.py` in the meantime).
- **Target** (after both fixes): `419f473e609ec0c3fd645a9dcb7ccd53a742fb8c22d17a45578e6cc6df2ff8fa`.
- **Admission**: a repo-wide search for the source hash found nothing live referencing it (one hit, in
  an already-archived `_archive/cleanup_2026-10-01.*/inventory.json` — not an in-progress corpus or
  checkpoint). Nothing is stranded under the old hash, so `ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` and
  `_BEHAVIOR_MIGRATION_SAMPLE_RATES` stay empty (the old hash is simply superseded, not retired — it is
  not added to `RETIRED_BEHAVIOR_HASHES` either, since inertness holds). Only the unconditional
  `RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY` pointer is bumped (that table is not a migration
  admission; `test_released_checkpoint_exception_is_separate_and_targets_this_build()` hard-asserts it
  always equals the live hash).
- **Inertness, measured**: `LinearAecProcessor.process_numpy()` run before and after both edits on
  three real double-talk clips from this campaign's allowed corpora — AEC 2021 blind
  (`aec_challenge_blind/doubletalk`, 16 kHz), AEC 2023 ICASSP blind
  (`aec_challenge_blind_icassp2023/doubletalk`, 48 kHz), and the tracked self-recording
  (`aec_record_{mic,ref}_10s.wav`, 16 kHz) — `error`/`echo_estimate` sha256 byte-identical on all three,
  both sample rates covered.
- **Harness-can-fail control**: constructed `AEC(AecConfig.from_preset(BALANCED))` and
  `AEC(AecConfig.from_preset(NO_FLOOR))` directly on the AEC-2021 clip above and confirmed their raw
  outputs differ (max abs diff 0.134, not byte-identical) — proves the byte-compare methodology used
  for the inertness check above is capable of detecting a real difference, so its "no difference"
  result is not vacuous.
- `AIAEC/tests/test_linear_aec_behavior_migration.py`: 18/18 passed against the new hash values with
  no test edits (the file compares against the live `aec_python_behavior_hash()` and the two named
  constants, neither of which needed to change).

## [Unreleased] — 2026-10-20 — Python<->C parity tolerances frozen (supersedes the Wave-S-prep provisional cut)

### Changed

- **`c_impl/test/pyc_parity/pyc_parity.py` (`TOL`) marked FROZEN instead of provisional**, with each
  of its nine constants re-justified against a full measurement on the real tree *after* the S-01 /
  S-03 / S-06 fixes below (44 scenarios, grids 8k256/16k256/16k512/48k1024, `--repeat-check`
  byte-identical both before and after this edit). The earlier numbers were set by the Wave-S prep
  workflow before those fixes existed; this freeze re-measures on the current tree instead of
  re-using that stale baseline.
- Five of the nine constants are **unchanged** because twice the worst currently-GREEN cell for that
  metric already rounds to the existing number (`L1_out_max_abs`, `L1_raw_max_abs`, `L1_out_rms_db`,
  `L2_out_max_abs`, `CTX_gain_max_abs`): `L1_out_max_abs`/`L1_raw_max_abs` 3.0e-4 (worst GREEN
  1.453e-4 / 1.153e-4, margin 2.06x / 2.60x); `L1_out_rms_db` -75.0 dB (worst GREEN -80.07 dB, margin
  5.07 dB); `L2_out_max_abs` 2.0e-3 (worst GREEN 7.893e-4, margin 2.53x); `CTX_gain_max_abs` 1.0e-3
  (worst GREEN 4.380e-4, margin 2.28x).
- Four constants **tightened** because the real margin was far wider than the old number implied:
  `L2_raw_max_abs` 2.0e-3 → **2.0e-4** (worst GREEN 9.049e-5, margin 2.21x); `L2_out_rms_db` -60.0 →
  **-61.0 dB** (worst GREEN -67.75 dB, margin 6.75 dB); `CTX_scalar_rel` 1.0e-3 → **2.0e-5** (worst
  GREEN 2.265e-6, the `erle_factor` scalar, margin 8.8x — kept well above the float32 one-ulp noise
  floor this harness's own docstring measures at ~1e-7, not frozen any tighter for that reason);
  `CTX_spec_rel_l2` 1.0e-3 → **1.0e-5** (worst GREEN 5.28e-7, the `echo_spec` block, margin 18.9x,
  same noise-floor caution). No constant was loosened.
- Verified no acceptance criterion moved: a per-cell diff of the pre-edit and post-edit full-suite
  JSON (all 44 scenarios x 2 layers) shows **zero verdict changes** (`PASS`/`EXPECTED_RED(S-xx)`/
  `INFO(...)` all identical), and `make test-header-fresh`'s equivalent static check
  (`s03_balanced_config_regen`) stays `PASS`. Both the pre-edit and post-edit full runs are
  individually `--repeat-check` byte-identical (sha256 `9eff612f…` before, `95024ee5f5…` after).
- **Gate can-fail proof**: in a scratch copy of this tree, an S-01-style mutation was planted in
  `src/pbfdkf.c` (the unconditional `call_counter` increment removed entirely, rather than an exact
  revert of the S-01 fix, so this is a stronger regression than S-01's historic gap and the numbers
  below should not be read as the size of that gap); against the frozen tolerances this moved
  `rec.16k256.bal` / `rec.48k1024.bal` / `dtb.16k256.bal` (among others) from `PASS` to
  `FAIL (layer 1 red, no marker)` with Layer-1 `out_max_abs` 0.12–0.22 (400-700x over the unchanged
  3.0e-4 tolerance) — 6 of 10 scenarios in that subset failed. The same subset against the real
  (unreverted) tree stays all `PASS`/`EXPECTED_RED`/`INFO`, as in the full run above.
- **Known gate blind spots this freeze does NOT close** (left open deliberately, not fixed on
  the side): (a) Layer 2's `INCONCLUSIVE_CHAOTIC` bucket absorbed **every** Layer-2 cell that
  exceeded tolerance in this measurement (7 of 44 scenarios; `layer2_red: 0` both before and after)
  — on the current default build, Layer 2 as a gate currently never fires `RED`, only `PASS` or
  `INCONCLUSIVE_CHAOTIC`; (b) `TOL` is still one global value shared by every scenario row — the
  measured per-row headroom on `L1_out_max_abs` alone ranges from 2.1x (`dtb.8k256.bal`,
  `rec.16k256.mild`, `recdc.48k1024.bal.nohpf`) to 1669x (`echo.8k256.bal`) under the frozen number,
  so a small deliberate constant change in a low-headroom row is far easier to catch than the same
  change in a high-headroom row; per-row tolerances remain out of scope for this freeze.

### Fixed

- **S-01: observer-path `call_counter` now increments unconditionally before the zero-mu early
  return.** `pbfdaf_process()` (`src/pbfdkf.c`) used to check `mu` against zero and return before
  bumping `call_counter` on an all-zero-mu hop; `filters.py`'s `_update_weights` always ticks its
  counter first. With the live observer enabled this desynced the observer's counter from Python's
  by one per such hop. The increment (floored at `LONG_MAX`, same as the equivalent counter in
  `erl_estimator.c`) now happens before any of the early returns. The production-default observer-off
  path is byte-exact unaffected: `make test-output-hash` still matches its golden digests on
  `ne10_neon` and `ne10_scalar`.
- **S-03: `include/aec_balanced_config.h` regenerated from its generator.** The checked-in header had
  drifted from `python/diag/gen_aec_balanced_config_h.py` by two literals (`_db_to_power_f32` moving to
  a float32 `powf`-based computation on the Python side without the header being re-run). The header
  is now byte-identical to the generator's output. Added `make test-header-fresh`
  (`c_impl/Makefile`): regenerates the header into a scratch copy of `python/` and diffs it against the
  checked-in copy, so a future drift between the balanced config and the generator fails a build gate
  instead of sitting unnoticed.
- **S-06: `epc_render_forced_remaining` now decrements every hop, unconditionally.** `aec.c`'s main
  loop used to tick this countdown only inside the RES-enabled block; `orchestrator.py` ticks it right
  after the shadow-rise tick, independent of `enable_res`/`return_res_context`. The field is
  diagnostic-only (nothing in the C library reads it back — it is set by the EPV/shadow-rise events and
  otherwise only decremented), so this keeps the counter's own value in step with Python's for
  diagnostics (e.g. the `pyc_parity` trace) with zero effect on any audio output.

All three were found and fixed together with the C<->Python function-map audit
(`docs/c_python_function_map.md`, added this entry); the map's own rows for `pbfdaf_process`,
`aec_post_run_observer_step`, `aec_process_core`, `suppression_gain_init` and `aecb_rate_cfg` have been
updated to describe the fixed behaviour instead of these three now-closed gaps.

## [Unreleased] — 2026-10-19 — C code tidy: NE10 is the only FFT backend, upstream-derived names removed from C, numeric `--preset` (no output change; C API renames)

Every `aec_wav` output, `AecResContext` field and `aec_get_mem_size()` value is bit-identical to the
commit before this entry (SIMD=1 and SIMD=0); the renames below change C symbols and file names only.

### Removed

1. **The KISS FFT backend.** `BACKEND` accepts `ne10` only (`BACKEND=kiss` is a Makefile error, as in
   `audio_common`); `LINK` is always `$(CXX)`. `test/golden/output_hash.txt` lost its `kiss` rows (the
   `ne10_neon` and `ne10_scalar` rows are unchanged). `test/run_fp_flag_policy_test.sh` and
   `test/run_counter_saturation_ubsan.sh` now pass `BACKEND=ne10`; `eval/run_c_ab.sh` builds and finds
   its binary under `bin/ne10-*`. `make audit-no-stdio` no longer has a "CALIBRATED" branch for a
   vendor residual: any stdio-family symbol in the linked executable is now fatal (the NE10 executable
   has none).
2. README / `docs/` statements that KISS is the production or a selectable backend; the pool-size
   tables in `docs/c_user_manual_zh_TW.md`, `docs/html/aec.html` and `docs/html/aec_c_api.html` now
   carry the NE10 measurements only.

### Changed

1. **C file names:** `delay_aec3.{c,h}` -> `delay_aec.{c,h}`, `aec3_post.{c,h}` -> `aec_post.{c,h}`,
   `aec3_scale.{c,h}` -> `aec_scale.{c,h}`, `aec3_balanced_config.h` -> `aec_balanced_config.h`.
2. **C identifiers** (mechanical, one old->new table, every new name checked for zero collisions
   across the repos): `DelayAec3` -> `DelayAec`, `delay_aec3_*` -> `delay_aec_*`, `Aec3Post*` ->
   `AecPost*`, `aec3_post_*` -> `aec_post_*`, `aec3_*` helpers of the scale module -> `aec_*`,
   `AEC3_*` constants -> `AEC_*` (the post driver's `AEC3_DELAY_ADJ_*` / `AEC3_DELAY_QUALITY_REFINED` ->
   `AEC_POST_DELAY_ADJ_*` / `AEC_POST_DELAY_QUALITY_REFINED`, because `AEC_DELAY_ADJ_NONE` already
   exists in `aec_state.c`), `Aec3BalancedRateDims` / `aec3b_rate_cfg` / `AEC3B_*` ->
   `AecBalancedRateDims` / `aecb_rate_cfg` / `AECB_*`, `DA_AEC3_BLOCK_SIZE` -> `DA_BLOCK_SIZE`,
   `SC_AEC3_STREAK_FRAMES` -> `SC_STREAK_FRAMES`, `ERL_AEC3_X2_MIN` -> `ERL_X2_MIN`,
   `pbfdkf_update_weights_aec3` -> `pbfdkf_update_weights`, `mu_aec3` / `scr_mu_aec3` -> `mu_aec` /
   `scr_mu_aec`, the config fields `use_aec3_echo_gen_window` / `use_aec3_residual_noise_gate` ->
   `use_aec_echo_gen_window` / `use_aec_residual_noise_gate`, and the test-support helpers
   `rde_update_aec3_strict` / `lazy_init_aec3` -> `rde_update_strict` / `lazy_init_strict`.
   No struct field was reordered, retyped or padded (the `sg_config.txt` digests and every
   `aec_get_mem_size()` value are unchanged). C callers outside this repo (the multi-lane pipelines) must
   follow the new names. The `a3_*` member prefix of `Aec` is kept.
3. **Comments in `.c` / `.h`** describe the algorithm instead of naming the upstream implementation; no
   `.c` / `.h` outside `test/historical/` contains `aec3`, `webrtc` or `kiss` any more.
   `python/diag/gen_aec_balanced_config_h.py` emits the new header name, guard and `AECB_*` / `aecb_*`
   names (its output differs from the checked-in header only by the two `SG_SPLIT_FLOOR_FAR_*` literals
   it already differed in before this change). `docs/timing_constant_inventory.{json,md}` cite the new
   file names.
4. **`--preset` takes `0|1|2` or the name** (`0` mild, `1` balanced, `2` aggressive, the existing
   `AecPreset` values) in `example/aec_wav.c`, `test/bench_rtf.c`, `test/parity_aec_e2e.c` and, through
   the new `python/preset_arg.py`, in `run_one_case.py`, `eval_aec_challenge.py` and `eval_manifest90.py`.
   The Python CLIs normalise back to the name, so `scores.json`, `AecPreset(...)` and the corpus contract
   see the same strings as before. `modules/**` (and its `--preset` in `orchestrator.py`) is untouched.
5. `parity_aec_e2e.c` header: tolerances are now documented against the maxima measured with the NE10
   backend (16 kHz: out 4.3e-3 against 2e-2; 48 kHz: out 6.2e-2 / 8.9e-2 against 1e-1; SIMD=1 and SIMD=0
   agree). The 8 kHz legacy-grid golden exceeds its 2e-2 ceiling (out max 1.3e-1); the retired KISS build
   gave the same figure, so it is a standing C/Python difference of that grid, not a backend effect.

### Fixed

1. A stale comment in `aec.c` claiming the saturation getter returns `double` (`saturation_detect()`
   returns `float`).

---

## [Unreleased] — 2026-10-19 — remove the Path-B backward-jump delay-change guard (measured net-negative; ABI: `sizeof(AecConfig)` −8 B, `aec_get_mem_size()` −16 B, raw `sizeof(Aec)` −8 B — see the ABI note below for why those last two differ)

### Removed

1. **`delay_backward_quarantine_enabled` / `delay_backward_quarantine_s`
   (Python/C `AecConfig`) and the Path-B delay-change guard they gated.**
   Introduced 2026-08-17 (see that entry further below for the original
   design and the pre-echo mis-lock it targeted). The mechanism held a
   delay-change candidate strictly EARLIER than the delay already in force,
   offered while the linear filter was still demonstrably cancelling
   (`aec_linear_is_cancelling()`), for a bounded window (default 1.0 s,
   converted to estimation cycles at construction), then accepted it
   regardless — delaying a sustained pre-echo mis-lock by one window,
   never curing it. Default OFF on both ports, so the shipped default path
   was always byte-identical to a build without the mechanism.

   **Measured net-negative: on an n=800 empirical counterfactual, the guard
   caught 0 bad relocks while creating 3** (genuine delay changes held back
   one window that should not have been). A guard with no demonstrated
   benefit and a measured cost does not ship.

   Removed: Python `AecConfig.delay_backward_quarantine_{enabled,s}`
   (`modules/config.py`); the construction-time `_delay_quarantine_hops`/
   `_delay_quarantine_left` counters, the `aec_reset()`-path clear, and the
   Path-B `_change_blocked` gate block in `modules/orchestrator.py`; C
   `AecConfig::delay_backward_quarantine_{enabled,s}` and
   `Aec::delay_quarantine_{left,hops}` (`c_impl/include/aec.h`); the
   matching default-init lines, the two `aec_validate_config` checks (its
   "toggles" tally drops from 16 to 15; the "scalar tunables" float tally
   was already stale at 23 against its own "22" label before this change
   and is accurate again at 22 after it — not touched, since no edit was
   needed), the `aec_carve()` seconds-to-cycles conversion block, the
   `aec_reset()` clear, and the Path-B `change_blocked` gate block in
   `aec_process_core` (`c_impl/src/aec.c`). Two now-dangling docstring
   mentions of the removed mechanism (`aec_apply_external_realign`'s and
   `aec_linear_is_cancelling`'s doc comments, both ports) are reworded, not
   deleted, since both functions keep other live callers. Deleted the two
   permanent regression tests that existed solely for this mechanism:
   `c_impl/test/test_delay_backward_quarantine.c` and
   `python/tests/test_delay_backward_quarantine.py` (7 tests; pytest's
   collected total drops from 312 to 305 with this removal and nothing
   else). Docs updated to match: `docs/aec_methods.md`,
   `docs/c_user_manual_zh_TW.md` §6.4/§9.4 (rows removed) and all of its
   `aec_get_mem_size()` byte tables (re-measured fresh against this
   revision, not patched from the old numbers — see the ABI note below),
   `docs/delay_estimator_design_zh_TW.md` §2 (mechanism write-up trimmed to
   the still-valid phenomenon description; see the preserved finding
   below), `docs/delay_estimator_productization_plan_zh_TW.md`,
   `docs/nn_integration_interface.md`, and
   `docs/html/{aec,aec_detail,aec_c_api,aec_c_api_params}.html`.

   **Preserved finding (not about this mechanism, discovered incidentally
   while comparing guarded-vs-unguarded builds in the now-deleted C test's
   multipath/path-addition scene).** At `g_old=0.5` (surviving old
   reflection and new, earlier path at equal gain) the matched-filter
   aggregator's dominant-peak vote is a near-exact 50/50 tie. Under the
   full-rate delay-estimation default (2026-10-18, `delay_est_period_s`
   0.5→0.0), the C test's two tried seeds split one each way while the
   Python reference scene's 20-seed sweep (per the now-deleted comment
   this is carried from -- the sweep itself left no reproducible script
   or test artifact in this repo; re-verify before relying on it again)
   landed on the newer path on every seed. Root-caused, not assumed: C's
   `da_argmax_i`/
   `da_argmax_incremental_update` (`delay_aec3.c`) and Python's
   `np.argmax`-based `_HighestPeakAggregator` (`lag_aggregator.py`) use the
   IDENTICAL tie-break rule (lowest bin index wins, same ascending
   bin-to-lag mapping); the pre-echo histogram on both sides is confirmed
   inert (never feeds the primary reported delay), so it cannot be the
   source. The asymmetry is two independently-seeded, differently
   distributed stochastic realizations (C: 32-bit xorshift, uniform
   [-0.35, 0.35], fixed seed `0x5EED1234`; Python:
   `np.random.default_rng(7)`, Gaussian ×0.1) of the same genuine
   near-50/50 dynamical tie — landing differently at `g_old=0.5` is exactly
   what two such realizations are expected to do, not evidence of a port
   gap. Full root-cause chain (the four-step exclusion that established
   this) preserved in `docs/delay_estimator_design_zh_TW.md` §2, since the
   test file that incidentally surfaced it is gone.

### Testing

Both backends build clean with `WERROR=1` (no new warnings). `make
test-output-hash`: 30/30 on both `BACKEND=kiss` and `BACKEND=ne10`, matching
this revision's golden exactly — the default path (`delay_backward_
quarantine_enabled` was already 0 on it) is bit-for-bit unaffected.
`pytest python/tests`: 305 passed, 0 failed (was 312 passed, 0 failed
before this change; the difference is exactly the 7 deleted tests).
`test-config-validation` (404), `test-external-realign`,
`test-process-context` (130), `test-duty-census` (17),
`test-delay-num-filters` (169), `test-runtime-preset` (66),
`test-rate-structural` (382), `test-reset-parity` (6), `test-zero-heap` and
`test-aec-res` (264) all still pass on both backends after the removal.
The A21 live-observer fid0 golden
(`SE/.research/2026-10-16/a21/tools/a21_c_port_fid0_e2e_golden.py` +
`parity_aec_e2e.c`, observer off and on) still PASSes after this change,
and the per-hop golden `.bin` files it produces are byte-identical before
and after this removal — the Path-B guard sat structurally adjacent to the
bad-relock soft-revert and live-observer code, so this was checked
explicitly rather than assumed from the output-hash gate alone (which only
exercises the default path).

**ABI note.** `sizeof(AecConfig)`: 228 → 220 B (−8 B: one `int` + one
`float`). Raw `sizeof(Aec)`: 6440 → 6432 B (−8 B, confirmed by compiling
both endpoints directly — NOT −16 B; a `cng_n2_windowed_psd`-sized
`Aec3PostConfig` field landed on an alignment boundary and grew the
embedded `Aec3Post post` member by +8 B, which cancels half of this
removal's own −16 B). Net effect by field position: fields between the
embedded `AecConfig cfg` member and the old `delay_quarantine_left`/`_hops`
location shift 8 B earlier; fields between that old location and `post`
shift 16 B earlier; fields after `post` shift only 8 B earlier (the
`Aec3Post` growth cancels the other 8). `aec_get_mem_size()` still drops by
exactly 16 B at every grid, every `delay_num_filters`, every mode
(`MATCHED`, `FIXED` at every `fixed_delay_samples`, `EXTERNAL_ALIGNED`) —
confirmed by `make test-delay-num-filters` on both backends — because that
figure is `ALIGN16(sizeof(Aec))`: 6440 rounds up to 6448 pre-change, 6432
is already a multiple of 16 post-change, so the *aligned* total drops 16 B
even though the *raw* struct only drops 8. `estimator_bytes` and
`ring_bytes` (`aec_get_mem_breakdown()`) are untouched by this change. Fresh absolute figures for every
grid/mode/n this project ships replace the old ones in
`docs/c_user_manual_zh_TW.md` §4 and in `docs/html/aec.html` /
`aec_c_api.html`; none of the previously recorded constants (the old
`sizeof(AecConfig)` = 204 B figure in particular) were still accurate on
this revision even before this change — this repo's struct layout had
already grown a `cng_n2_windowed_psd` field and live-observer/soft-revert
state since those numbers were last measured, so they were re-derived from
scratch rather than patched.

---

## [Unreleased] — 2026-10-19 — feed the CNG N2 tracker a windowed capture PSD (new default-OFF flag)

AEC3's ComfortNoiseGenerator estimates its noise target from Y2 formed via
`WindowedPaddedFft(kSqrtHanning)` — the same windowed FFT the subtractor
forms E2 from, not a rectangular one. This port's N2/N2_initial tracker was
fed from `near_psd`, the rectangular `|near_spec|^2`, so its target
coordinate ran ~3 dB high relative to AEC3's windowed reference. This is a
distinct, upstream bug from the deleted `cng_window_power_compensation`
flag below, which patched the symptom on the CNG *output* side even though
the window-loss compensation in the synthesis/injection path (the
`kSqrt2Sin`-style scaling) is already correct; the real fix belongs at the
tracker's input.

Adds `cng_n2_windowed_psd` (default OFF, Python + C, mirrored), which
re-points the N2/N2_initial tracker (first-hop init + per-hop EMA) at the
windowed capture PSD already shared with `erle_windowed_capture_psd` —
reusing the magnitude `aec3_post_run`'s Step 2 already computes
unconditionally each hop, so no second FFT. The ERLE consumer stays gated
on `erle_windowed_capture_psd` alone.

Isolated seeded-N2 A/B harness (scratch, not committed): the tracker's own
absolute level shifts by −2.73/−2.39 dB (grid A/B) when the flag goes ON,
closely matching an independent single-hop-free witness of the
rectangular-vs-windowed PSD ratio (−3.00/−3.02 dB, averaged over 50 hops).
Flags-off bit-exact confirmed two ways: C's existing test suite
(`test_output_hash` 30/30, `test_aec_res` 264/264, `test_config_validation`
404/404, `test_reset_parity` 6/6, `test_zero_heap`) unchanged, and a
byte-exact Python A/B run of `fileid_0` against the pre-change code. Quick
sanity (NOT a full validation pass): fid0 standard metrics finite/stable
both ways; 14/15 FS Rule A clips pass, the one flagged traced to a
near-floor CN-level transition (sub −44 dBFS) — the exact, in-scope
mechanism this flag changes, not a new echo-leak path.

**NOT corpus-validated and NOT a promotion candidate**: CN level affects
AECMOS deg/echo scores and this project's 800-case gate is required before
any default-on decision, which is explicitly out of scope here.

---

## [Unreleased] — 2026-10-19 — delete three default-off mechanisms that never became viable

**`tail_evidence`** (`tail_evidence_enabled`/`tail_evidence_feed_unbounded`
and the other `tail_evidence_*` fields, `python/modules/residual/
tail_evidence.py`): superseded by `live_observer_enabled`, which closes
55%+ of the achievable ceiling against `tail_evidence`'s 24-30%. No test or
cross-repo references.

**`cng_window_power_compensation`**: built on a misdiagnosis. The
comfort-noise synthesis window's OLA power loss is already compensated by
`kSqrt2Sin`, matching AEC3's own design (see the unchanged comment on
`_aec3_sqrt2_sin_lut`). The dB gap this flag chased is explained by the N2
tracker's min-statistics bias and this port's near-PSD/Y2 coordinate
offset (see `cng_n2_windowed_psd` above) — neither of which this flag
corrected; turning it on overshot by about +3 dB. No C port, no
measurement script in committed main, no test references.

**`delay_acquire_protect_inst_erle`**: a more conservative alternative to
the kept, default-ON `delay_acquire_warm_transfer` (block realign outright
during delay acquisition instead of warm-transferring the taps). Never
became competitive on the 800-case battery and has been inert in both
languages. `delay_acquire_warm_transfer` and `delay_acquire_inst_erle_db`
are untouched and remain live production code.

All three were already OFF by default (zero production behavior change).
Verified: C `test-output-hash` digest and its checked-in golden are both
unchanged; full `c_impl` test-* suite and `make selftest` stay green; 312
pytest cases pass before and after; and a direct before/after balanced
16 kHz end-to-end golden (`python/diag/gen_aec_e2e_golden.py`, 4200 hops)
is byte-for-byte identical, confirming the Python orchestrator changes
are bit-exact. Removing `delay_acquire_protect_inst_erle` shifts every
`AecConfig` field after it; `Audio_ALG` embeds this struct by value and
needs a recompile when its `lib/aec` pin advances past this commit (no
functional reference to the field was found there).

---

## [Unreleased] — 2026-10-09 — standalone RES stage (`aec_res.h`), output-hash and NEON-twin gates

### Added

- **`aec_res.h` / `src/aec_res.c`: the residual-echo suppression gain as a
  standalone stage.** `aec_res_get_mem_size` / `aec_res_init` /
  `aec_res_reset` / `aec_res_process` / `aec_res_set_preset` /
  `aec_res_split_floor` / `aec_res_n_bins` / `aec_res_hop_size`, plus
  `aec_cng_fill` (the comfort-noise fill of the multi-lane pipelines). For
  integrations that fuse several linear AEC outputs and need one RES gain on
  the fused spectra. Caller-owned memory, no heap, no stdio. The suppressor is
  configured by the same code a lane uses, so equal `AecConfig` gives
  bit-identical configuration. See `docs/html/aec_res.html` and
  `docs/c_user_manual_zh_TW.md` §8.6.
- **`make test-output-hash`.** Hashes the emitted hop and every
  `AecResContext` / `AecLinearContext` field over a seeded 6000-hop scene for
  3 grids x shadow on/off x {`aec_process`, `aec_process_context` levels 0, 1,
  2} against checked-in digests per FFT build (`kiss`, `ne10_neon`,
  `ne10_scalar`); also checks that the gain-path-free digest does not depend on
  the context level.
- **`make test-neon-twins`.** NEON-vs-scalar equivalence for the first-maximum
  tie rule of `rsa_argmax_first` and the hold-counter boundary of
  `decrease_erle_for_low_render`, which the output hash cannot see.
- **`make test-aec-res`.** `aec_res` against the library and the multi-lane
  pipelines' own code: the suppressor configuration equals a heap lane's and a
  caller-pool lane's and the checked-in digests on every supported grid and
  preset; the gain is bit-identical per hop over long randomised runs against
  a transcription of the pipelines' RES stage (preset retarget and reset
  included); `aec_cng_fill` gives bit-identical spectra and generator state;
  argument checks, sizing, alignment, no write outside the block, reset and
  preset semantics; plus constructed cases the transcription cannot reach: a
  render block straddling the split-floor latch power (pins the exact 32768
  scale), the initial-state exit with the two initial-phase flags cleared, and
  the first recovery hop after a saturated echo (pins `floor_first_increase`).

### Fixed

- **Average-render reverb read the newest partition as its "one-older" slot.**
  The ERLE render power `X2[delay] + reverb(X2[delay + 1])` is read from the
  filter's own `n_partitions`-slot ring, so the delay (in partitions) can be at
  most `n_partitions - 2` for `delay + 1` to stay older than the newest slot.
  The C library capped it at `n_partitions - 1`: with the direct-path delay at
  the last partition the "past" slot wrapped onto the newest one and fed the
  reverb model the current block. The Python reference and AEC3 (whose ring is
  longer than any delay) never alias. The slot arithmetic is now one function,
  `aec3_post_avg_reverb_slots()`, with the cap `n_partitions - 2`; it differs
  from the previous output only on hops where the delay estimate is at or beyond the
  last partition. `make test-output-hash` is unchanged on `kiss`, `ne10_neon`
  and `ne10_scalar` because the seeded scene never reaches that delay.
- `make test-avg-reverb-slots` (`test/test_avg_reverb_slots.c`) checks the
  slots against a step-by-step walk for ring sizes 2..9, every write position
  and delays from -3 to beyond the ring length, and that the past slot never
  aliases the newest one. It is part of the C-side release gates.

### Changed

- `aec_build_sg_config()` (internal, `aec_config_build.h`) now builds the
  `SuppressionGainConfig` / `SuppressionGainTuning` that `aec_carve` built
  inline. No change to library output (`make test-output-hash` unchanged).
- `test/test_zero_heap_aec.c` and `test/no_stdio_main.c` also exercise `aec_res`.
- Archived A/B evidence: the harness copy and source diffs whose headers named
  build-host directories, and author addresses in two archived diffs, were
  removed from the tree; per-case scores and READMEs remain.
- `test/test_output_hash.c` lists the constants its digests cannot see
  (`floor_first_increase`, the initial `Q_high`/`Q_low`/`Q` of the main filter,
  `alpha_r`, the fullband ERL step) and the gate that covers each.
- Comments in the library sources, headers, example, tests and the Python
  reference are release-style: development history and internal tags are
  removed, contracts and numerical reasons stay. No code change (the token
  streams of every C file are identical, except two printed message texts in
  the tests).

### Performance (bit-identical; no behaviour change)

Five hot spots now run as NEON, each with a scalar twin of identical
arithmetic (`-ffp-contract=off`, no estimate instructions, divides stay
divides). Output is byte-identical except the payload bits of a NaN result:
`make test-output-hash` (30 rows) is unchanged on `kiss` and `ne10` x
`SIMD=0/1`, and a whole-output hash of 54 further runs (3 grids x shadow on/off
x entry points, 6000 hops each, three scene seeds) is identical on five build
variants. Only `da_resample48_pair` can return a NaN with another payload than
the scalar chain (about 16 % of the calls of a random-bit-pattern differential
test); the four `sk_` kernels returned identical bits in every call of that test.

- `da_resample48_pair` (`delay_aec3.c`): the capture and render 48 kHz
  anti-alias cascades, two dependent chains of float32 multiply-adds, run as the
  two lanes of one vector. Only the 48 kHz grid is affected: -12.7 % full,
  -20.1 % linear-only.
- `sk_max_abs_f32`: the hop peak magnitude of the echo estimate (main and
  shadow filter) and of the render block: -1.3 to -1.5 % full, -2.4 to -3.2 %
  linear-only on all three grids.
- `sk_noise_floor_track_f32`: the render noise floor and hold counter of the
  residual echo estimator: -1.6 to -1.7 % full (the post stage; no change
  linear-only).
- `sk_raise_to_f32`: the per-bin lower bound on the suppression gain: -1.6 to
  -1.7 % full (no change linear-only).
- `sk_stationary_mask_u8`: the stationary-bin mask and count: -2.0 to -2.2 %
  full, -1.5 % linear-only.

Per-hop cost of `bench_rtf` (`balanced`, comfort noise on, speech-like scene,
`--stage full|linear`), median of 30 interleaved runs, one core of the dev
machine (Apple M4, NE10), microseconds per hop:

| grid (rate / FFT) | full before -> after | linear-only before -> after |
|---|---|---|
| 16 kHz / 256 | 8.83 -> 8.24 (-6.7 %) | 5.08 -> 4.87 (-4.1 %) |
| 16 kHz / 512 | 18.10 -> 16.84 (-7.0 %) | 10.86 -> 10.43 (-4.0 %) |
| 48 kHz / 1024 | 39.50 -> 32.22 (-18.4 %) | 24.75 -> 18.88 (-23.7 %) |

The per-kernel figures above are each measured against the build without that kernel change.

### Added (benchmark and tests)

- `bench_rtf` takes `--sr`, `--fft`, `--stage full|linear|both` and
  `--scene noise|bursts` (a 12 s speech-like scene with far-end bursts, double
  talk and a level step), reports the median pass next to the fastest, and with
  `PROFILE=1` prints the mean stage buckets.
- `test_delay_kernels` checks the 48 kHz pair kernel against two single-channel
  calls across carried state, odd lengths, output-cap truncation and the
  single-call fallbacks; `make selftest` covers the four new kernels, and
  `sk_max_abs_f32` and `sk_stationary_mask_u8` also against the loop each stands
  for, written out in the test.
- Kernel gates: `make selftest`, `make test-output-hash`, `make test-aec-res`,
  `make test-neon-twins` and `make test-delay-kernels` form the gate list for
  these kernels, and `test-delay-kernels` stays in it because it is the only gate
  that sees the 48 kHz anti-alias pair kernel: one unit in the last place of the
  filtered stream does not move the integer delay estimate, so no
  `test-output-hash` digest changes, and `selftest` covers the `sk_` kernels of
  `aec_simd_kernels.h`, not `delay_aec3.c`. With `SIMD=0` that test compares the
  scalar chain with itself. The selftest's `sk_stationary_mask_u8` cases include
  every bin set up to n = 65549: they are the only gate for the byte-accumulator
  fold limit (255 steps of 16 bins) and, with `test-output-hash`, a gate for the
  lane sum.

## [Unreleased] — 2026-10-06 — linear-half context level (`AEC_CONTEXT_LINEAR_ONLY`) and NEON hot spots, all bit-identical

### Added

- **`AecConfig.spatial_linear_context` level 2 (`AEC_CONTEXT_LINEAR_ONLY`).**
  The field stays an `int` (so `sizeof(AecConfig)` and every pool size are
  unchanged) and now takes 0 (full context), 1 (this lane's own `res_gain` is
  not computed -- the existing meaning) or 2. Level 2 is for a caller that
  reads only the linear half of `AecResContext` / `AecLinearContext`
  (`formed_hop`, `error_spec`, `echo_spec`, `near_spec`, `far_spec`, the delay
  and verdict fields): the post block stops after the AecState update,
  skipping the reverb/residual estimator, the stationarity R² zeroing,
  comfort-noise tracking and the suppression gain. None of those write
  anything the linear filter, the AecState, the stationarity estimator, the
  avg-render reverb or the pending echo-path flags read, so every linear
  output is byte-identical to levels 0 and 1 (`test_process_context`: 3 grids
  x shadow on/off x level 0/1, 600 hops each, with the `usable_linear`
  verdict toggling during the run; skipping the AecState update as well makes
  that test fail). `res_gain`, `r2` and `comfort_noise` read `NULL` and
  `res_floor_protect` reads 0. Same validity rule as level 1
  (`enable_res=0 && return_res_context=1`); values outside 0..2 are rejected.
  Measured on one core of the dev machine, `aec_process_context()` 7.31 ->
  6.32 us/hop (16 kHz / 256), 15.85 -> 13.98 (16 kHz / 512), 31.89 -> 27.87
  (48 kHz / 1024), shadow off; 11-14 % in all six grid x shadow cases.

### Performance (bit-identical; no behaviour change)

Eight scalar or latency-bound hot spots of the linear path now run as NEON,
each checked against its previous implementation (differential fuzz where the
function is directly callable, a 12-configuration whole-AEC output hash --
3 grids x shadow on/off x `aec_process`/`aec_process_context` over 6000 hops
-- for all of them together, and scalar-build == NEON-build):

- `rsa_update`: the narrow-band counters (int64 lanes), the peak search
  (two-pass first-index argmax, scalar walk on any non-finite maximum) and
  the render-block max-abs.
- `saturation_detect`: both integer counts.
- `subband_erle`: the accumulated-spectra update and the per-band hold /
  release loop.
- `stationarity_estimator`: the flag compute, the hangover update and the
  neighbour smoothing (16 flags a step, original left neighbour carried).
- `pbfdkf_h_error_refresh`: the per-bin leakage update and the clip.
- `sk_cmac_np_ring_f32` (new, `aec_simd_kernels.h`): the partitioned echo
  accumulation keeps each four-bin accumulator in registers across the
  partitions; same fma-form product, same partition order.
- The render and capture decimators run as the two lanes of one vector
  (`da_decimator_decimate_pair`): same products, same grouping, no fused
  multiply-add. `test_delay_kernels` gained a state-carrying pair sweep
  against two generic walks.

Measured on one core of the dev machine, `aec_process_context()` (shadow on,
16 kHz / 256, 16 kHz / 512, 48 kHz / 1024): 9.02 -> 7.64, 18.23 -> 15.52,
38.78 -> 34.46 us/hop (-15 %, -15 %, -11 %), on top of level 2's 11-14 % for
callers that read only the linear half.

## [Unreleased] — 2026-09-22 — validated DT floor and cross-port fixes

### Changed (measured output trade-off)

1. The default double-talk RES floor is `-20 dB` instead of `-16 dB`.  This
   does not change the latch or pretend to repair near/far classification; it
   chooses a different point on the existing floor trade-off.  On the paired
   832-file set, `-20` versus `-16` improves far-end-single-talk echo MOS by
   `+0.053/+0.064` and ERLE by `+0.46/+0.53 dB` (movement/static), while
   double-talk degradation MOS changes by `-0.027/-0.030`; near-end-only is
   neutral.  The same gain also controls comfort-noise injection, so the
   release evidence includes both AECMOS and raw ERLE/band-energy measures.
2. Python's dB-to-power floor conversion now uses explicit float32 operands,
   matching C `powf`.  The former Python binary64 expression differed by one
   ULP for `-16/-28/-38 dB`; the conversion is initialization/runtime-preset
   control work, not a per-hop cost.

### Changed (same policy with the defaults)

3. The double-talk floor decision moved inside `aec3_post_run()`, immediately
   before its suppression-gain consumer. C and Python still use the existing
   near-recent latch policy; the move makes one site own and export the exact
   per-hop decision. Apart from the selected `-20 dB` value, the RES output is
   byte-identical to the previous policy (90 blind stems x 256/512 grid x CNG
   on/off, 360 renders, 0 differ).
4. `AecResContext` adds `res_floor_protect` (that per-hop floor decision, for
   fused consumers to OR across lanes) and `usable_linear` (this hop's
   usable-linear-estimate verdict: with 0 the exported `r2` is the conservative
   nonlinear estimate, not one derived from the linear echo estimate). Additive
   public C-struct ABI change; applications and static pipeline pools must be
   rebuilt together.

### Fixed

5. The shadow-copy warm-up was a fixed 50 hops, so it spanned 400--800 ms over
   the supported grids although its last validated operating point was 500 ms.
   Both ports now derive the hop count from 500 ms using the shared float32
   timing rule (31/62/31/47 hops at 8k/128, 16k/128, 16k/256 and 48k/512).
   The paired 360-render policy check remains byte-identical.

---

## [Unreleased] — 2026-09-21 — fix adaptive reverb-decay units

### Fixed

1. The optional partition-energy reverb-decay estimator stored its fitted
   slope per sample even though the residual-echo consumer retimes an AEC3
   per-4-ms-block value. This made the estimate about 64 times too sticky at
   16 kHz and sample-rate dependent. The estimator now converts each fitted
   partition slope to the 4 ms convention using the live sample rate and hop.
   Parameterized tests cover 8/16/48 kHz at 8/16 ms hops; the C parity-support
   implementation uses the same 64-sample convention on its fixed 16 kHz grid.

### Compatibility

2. `use_adaptive_decay` remains disabled by default, so the shipped fixed-decay
   processing path and public API are unchanged.

---

## [Unreleased] — 2026-09-04 — retime the shadow-copy baseline and discard abandoned-filter evidence

### Fixed

1. **The shadow-copy error baseline was still an EMA in hops.** The
   `0.995 old + 0.005 new` copy-error baseline encoded an approximately 2 s
   pole on the former 160-sample/16 kHz grid. Both ports now derive the
   retention from each instance's hop/sample-rate grid (0.996 at 8 ms,
   0.992 at 16 ms, 0.9947 at 48 kHz). The sibling `HYS_STREAK_MIN=10` gate
   stays a 10-hop count on every grid: retiming it to 100 ms (12 hops at the
   shipped 16 kHz/128 grid) was measured on the 90-case blind subset and
   broke one near-end single-talk clip with a loud, echo-free reference (deg
   MOS 2.33 -> 1.54, output 5 dB under the capture for 26 s) because the
   rescue streak stopped completing; 11 hops fails the same way and 10
   restores it. The retention change alone is neutral on that clip and on
   the subset (far-end echo MOS +0.006, double-talk within 0.004). Tests
   pin, per product grid, which hop triggers the copy and which coefficient
   the EMA actually applies.

2. **C internal delay recovery retained three decisions derived from taps it
   had just abandoned.** `poor_coarse_counter`, `coarse_reset_hangover` and
   `leakage_div_sustained_counter` were already cleared by Python's shared
   filter-latch reset and by C external realignment, but not by C's
   delay-first/delay-shift hard restart. They now live in C's common latch
   reset as well, so stale pre-reset evidence cannot fire a recovery decision
   later in the same hop.

### Compatibility

3. **No public processing API or static-pool layout changes.** C retires the
   redundant shadow `streak` counter and stores the per-grid EMA retention in
   its slot; the existing copy counter carries the equivalent max-threshold
   decision, so `sizeof(ShadowCopy)`, `sizeof(Aec)` and every pool offset
   stay unchanged. The pre-multirate `shadow_copy_init()` entry point remains
   as a legacy-grid wrapper around the grid-aware initializer: it keeps the
   authored 0.995 retention, with the EMA weighting the new sample by
   `1 - retention` instead of the former `0.005` literal.

4. **The AIAEC 16 kHz behaviour hash moves and old `linear_error` must be
   rematerialized.** This is a live retune, not a hash-only refactor: on
   AIAEC's frozen 512/256 grid the baseline retention becomes 0.992 instead
   of 0.995, which moves `error_is_normal` and therefore the copy decisions,
   and the C hard restart now clears evidence it used to keep. The old
   identity therefore cannot enter AIAEC's byte-equivalent migration table.
   Existing mic/far/target WAVs may be kept; recompute only `linear_error`,
   repack, and retrain checkpoints that consume that channel.

---

## [Unreleased] — 2026-08-30 — preserve FilterAnalyzer's IR exclusion span at 48 kHz

### Fixed

1. **The ConsistentFilterDetector peak-exclusion window was frozen in
   16 kHz samples.** Its 64-sample low side and 128-sample high side encode
   4 ms / 8 ms at the rate where AEC3 and this port were validated. At
   48 kHz the literals narrowed the window to 1.33 ms / 2.67 ms, changing the
   impulse-response floor and secondary-peak test on a default-ON path.
   Python and C now use 192 / 384 at 48 kHz and retain 64 / 128 at both
   16 kHz grids. The legacy 8 kHz grid is outside this correction's release
   scope and also retains its established 64 / 128 values.

2. **No static-pool or processing-API change.** The C implementation stores
   `sample_rate` in the former duplicate `FilterAnalyzer` threshold slot; the
   live threshold remains in `FaConsistentDetector`. Consequently
   `sizeof(FilterAnalyzer)`, `sizeof(Aec)`, static-pool offsets and public
   process entry points remain unchanged. Exact effective-value tests cover
   all supported grids in both ports; the 48 kHz structural paths remain
   finite and the two 16 kHz grids retain their previous numeric window.

### Compatibility

3. This is an intentional audio-path correction at 48 kHz, so newly
   materialized 48 kHz `linear_error` data must
   record the new frontend behavior hash. Existing 16 kHz data may be carried
   forward by AIAEC's rate-scoped migration gate: the exact
   recorded-hash-to-current-hash pair is backed by measured 16 kHz byte
   equality. The gate explicitly rejects old 48 kHz material, which must be
   rematerialized with this frontend.

---

## [Unreleased] — 2026-08-26 — `aec_reset()` returns a fresh instance, as its contract always claimed (BEHAVIOUR CHANGE after a reset; AIAEC behaviour hash moves)

### Fixed

1. **State survived `aec_reset()` that a fresh construction never has.** The
   header promised an instance indistinguishable from a newly built one on the
   same config, and the very first post-reset hop already diverged from that
   twin. Two of the three culprits sit behind the same structural cause: the
   filter wipes `aec_reset()` calls — `pbfdkf_reset()` / `pbfdaf_reset()` —
   are ALSO the mid-stream delay-relock wipe, and that caller re-arms the
   AEC3 startup gates itself at each of its own call sites
   (`pbfdkf_handle_echo_path_change()` for the refined filter, the explicit
   shadow counter writes beside it). What the filter functions leave standing
   is exactly what the relock path is about to overwrite — but `aec_reset()`
   had no such call site, so it inherited a warm filter's gates.

   `pbfdaf_reset_carried_state()` / `pbfdkf_reset_carried_state()` restore
   what those functions carry forward, and only `aec_reset()` calls them, so
   the relock path is unchanged by construction rather than by argument. They
   cover the startup/excitation/saturation gates (`call_counter`,
   `poor_excitation_counter`, `saturated_capture`, `block_stationary`), the
   InitialState tracking triple, the round-robin constraint cursor, the
   last-hop `SubtractorOutput` peak, the four orchestrator-fed refresh inputs
   on PBFDKF, and its per-bin `H_error` — which `pbfdkf_rearm_kalman()` never
   touched, because the relock path gets it from the
   `handle_echo_path_change(delay_change=1)` that follows it.
   `poor_excitation_counter` is restored to `aec_carve()`'s rate-scaled value,
   not to the reference-grid literal the init path writes before `aec_carve`
   overwrites it.

   Two more, outside the filters: the inst-ERLE slope ring
   (`erle_slope_len`/`_head`, which `aec_reset_filter_latches` already
   emptied and `aec_reset()` did not, so `erle_slope_max_recent()` read the
   previous run's history as this instance's own), and the `AecResContext`
   stash (`last_far_power`, `last_shadow_dt`, `last_dt_indicator`,
   `last_is_stationary_dt`) — LAST-HOP values published on a seam an external
   RES reads, on an instance that now has no last hop.

2. **Deliberately kept.** `delay_generation` still BUMPS rather than zeroing:
   its contract (aec.h) is an alignment token external pollers compare against
   a cached value, and a reset that reused a generation a consumer had already
   seen would be indistinguishable from no reset at all. The delay
   estimator's clockdrift tracker stays untouched for the reason
   `delay_aec3_reset()` already states — a 30 s hardware-drift estimate is not
   reset-scoped audio history. `constraint_round_robin` is config, not state.
   `Aec3Post::trace` is now restarted, but in `aec3_post_reset()` rather than
   in `aec_reset()`: it is written by that block and read only by
   `aec_get_debug_status()`, never by the audio path, so restarting it moves
   no sample on either caller and stops a `--debug-trace` row printed after a
   reset from attributing a pre-reset hop to a block that has not run since.

3. **Python twin, and one gap the C side did not have.** `filters.py` grows
   the same `reset_carried_state()` split, and `AEC.reset()` calls it. Porting
   the C reset field-for-field surfaced a divergence in the other direction:
   the near-end recency pair `_ne_above` / `_ne_recent_frames`, which
   `aec_reset()` has always cleared, survived `AEC.reset()` and kept steering
   the soft-recovery branches of the next run. It is the only residue in
   either port that moves the FIRST post-reset hop at 48 kHz/1024, and the
   only one a 16 kHz-only gate would have missed.

   The construction-time excitation-counter seed now lives in one place
   (`AEC._seed_poor_excitation_counters()`), called by `__init__` and
   `reset()`, so the value a reset restores cannot drift from the value
   construction wrote. Unlike the C twin, whose `aec_carve()` writes both
   filters unconditionally, this port only seeds the counter when a
   RenderSignalAnalyzer exists — which is why the restore is the
   orchestrator's and not the filter's.

### Removed

4. **`PBFDAF.power` / `PBFDAF.alpha_power` (Python).** The far-end power EMA
   and its retimed coefficient had exactly one reader in either port — their
   own cold-start guard, whose decision cannot depend on the value — and the C
   twin dropped them already. The Python side now follows, along with the
   cold-start branch that was the array's only consumer.
   `docs/timing_constant_inventory.md` entry `pbfdaf-alpha-power` records the
   removal in both ports; the entry stays closed rather than being deleted,
   since it is what answers the audit question for a constant that once
   existed and it carries the A/B evidence that retired it.

### Testing

`make test-reset-parity` (new, both backends): a never-warmed instance and one
warmed 600 hops on a different echo path then reset must produce byte-identical
`out[hop]`, `AecResContext.formed_hop` and `AecResContext.error_spec` over 400
post-reset hops, at 16 kHz/256, 16 kHz/512 and 48 kHz/1024, with RES enabled
and context-only. 6/6 green under both `BACKEND=kiss` and `BACKEND=ne10`; 0/400
hops differ in every case, against 399/400 from hop 1 before the fix.
`python/tests/test_aec_reset.py::ResetEqualsFreshInstanceTests` is the same
property on the Python port, over the same three grids — the grid set is
load-bearing there, since `H_error` residue only steers a decision at the two
larger transforms.

Mutation-proved by backing out one restore at a time and confirming the gate
goes red: `call_counter` (hop 1), the InitialState triple (hop 9),
`partition_to_constrain` (hop 8), `H_error_per_bin` (hop 5), both
`reset_carried_state` calls together (hop 1), and on the Python side the
near-end recency pair (hop 0). Six restores stay green when backed out
individually and that is the correct result, not a weak gate:
`saturated_capture`, `block_stationary` and `last_s_max_abs` are written every
hop before any consumer reads them (their own field comments and
`aec_reset_filter_latches`'s say so); `e2_coarse_for_refresh` /
`e2_coarse_per_bin_valid` likewise; `poor_excitation_counter` was measured at
500 on both sides of the reset, since the runtime increment ceilings at
`n_partitions` and the `< n_partitions` gate cannot tell the two apart; and the
inst-ERLE slope ring is read only by the delay Path-A `already_cancelling`
guard, which the gate's scene does not reach. They are restored because the
contract is "a fresh instance", not because a sample moves today.

Streaming without a reset is unchanged: `aec_wav` output is byte-identical to
`d5193ad` on 2 captures x 2 configs x 2 backends (`cmp`-verified against
distinct binaries), and the Python port is byte-identical to the same baseline
on 2 captures x 2 grids — which is also the measurement behind the dead-field
removal above. `parity_aec_e2e` PASS at 16 kHz (balanced, mild, aggressive,
and `--no-res`) and at 48 kHz. Full per-target suite green on both backends
with `WERROR=1`, plus `audit-profile`, `audit-no-stdio`,
`test-no-testing-symbols`, and 288 Python tests.

No ABI change: `sizeof(Aec)` stays 5,848 B, `sizeof(PBFDKF)` 504 B,
`sizeof(PBFDAF)` 336 B, and `aec_get_mem_size()` is identical at 16 kHz/256,
16 kHz/512 and 48 kHz/1024 — this clears state, it does not add fields.

### Impact

Any caller that reuses an instance across streams gets a different second
stream: previously it inherited a converged filter's startup gates and Kalman
`H_error`, which is why the first post-reset hop already differed from a fresh
twin. That is the fix, not a side effect. A caller that constructs per stream
sees nothing.

The Python signal path changed, so the AIAEC behaviour hash moves. Dataset
generation builds a fresh engine per sequence and never calls `reset()`, and
nothing reads the removed fields, so the materialized `linear_error` is
byte-identical across the pair — an already-rendered corpus needs no
rematerialization and travels forward on the sanctioned
`ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` entry in the integration repo. An
INTERRUPTED rematerialization is keyed on the raw-text `fingerprint()` instead
and restarts regardless; completed sequences are unaffected.

## [Unreleased] — 2026-08-26 — the duty-cycle ERLE watchdog leaks by the clock, not by the hop (BEHAVIOUR CHANGE at 8 kHz/256, 16 kHz/256, 16 kHz/512 and 48 kHz/1024)

### Fixed

1. **The watchdog's leaky ERLE peak decays 0.1 dB/s at every grid.** The
   delay-estimator duty machine resumes full-rate matched-filter analysis when
   `erle_windowed` falls more than 6 dB below a running peak, and that peak
   leaks so a stale high-water mark cannot hold the machine at full rate
   forever. The leak was a bare `0.001f` per hop — 0.1 dB/s only on the
   legacy 10 ms grid it was calibrated on. On the shipped grids it read
   0.125 dB/s at 16 kHz/256 (25% too fast, so an ordinary quiet stretch of
   far-end bleeds the peak down and looks like a collapse), 0.0625 dB/s at
   8 kHz/256 and 16 kHz/512 (a real echo-path change stays in decimated
   analysis about twice as long), and 0.09375 dB/s at 48 kHz/1024.

   `Aec::duty_erle_leak_db` now carries it, converted once at carve time from
   the hop. The machine's other two spans were already grid-derived —
   `hold_hops` from `delay_est_init_s`, `K` from `delay_est_period_s`, both
   recomputed from the hop where they are read — so this leak was the only
   quantity in the block still frozen at a hop count.
   The conversion is linear in the hop period, not a power law: an additive
   leak sums over N hops, it does not compound. At hop=160/sample_rate=16000
   the conversion is the identity and returns `0.001f` bit-for-bit, so the
   grid the constant was calibrated on is untouched by construction rather
   than within a tolerance.

### Impact

This changes WHEN the watchdog fires, which changes which hops the matched
filter is engaged on, which can change the delay the estimator reports. It is
a behaviour change on all four shipped grids — none of them has a 10 ms hop
(16.000 / 8.000 / 16.000 / 10.667 ms), so the frozen literal was wrong on
every one of them and there is no grid where old and new coincide.

Measured on 832 AEC Challenge blind stems (far-end singletalk, doubletalk,
near-end singletalk) at both 16 kHz grids, balanced, RES enabled — 1,664
streams:

| what moves | streams |
|---|---:|
| leaky-peak trajectory | 1,132 (68.0%) |
| realised duty census (`duty_hops_run`) | 62 (−30 to +35 hops) |
| final delay estimate | 1 |
| output samples | 2 (0.12%) |

Both output-moving streams are at 16 kHz/256; none at 16 kHz/512. The two:
`OjdIdZgJDk6hLAQL07KORA_farend_singletalk` diverges at 16.31 s when the delay
settles on 584 samples instead of 564 (24.5% of hops differ afterwards,
max |Δ| 0.16), and `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk` at 28.54 s from a
26-hop delay excursion that recovers to the same final value (27.1% of hops,
max |Δ| 0.78). Everything else in the sweep is byte-identical, because the
leak only changes a DECISION when the peak is armed (>6 dB) and a collapse
lands within a fraction of a dB of the 6 dB edge — on one worked example the
old leak had bled the peak to 8.16 dB against a 2.29 dB ERLE (a 5.87 dB
collapse, no trip) where the retimed one still read 8.33 dB (6.04 dB, trip).

The two reference captures used for seam evidence throughout this changelog —
the `0I0XMl3M0ECO0U1N0cJvpg` doubletalk clip and the 10 s pair in
`wav/aec_record` — are byte-identical on output hop, `formed_hop` and
`error_spec`, at 16 kHz/256 and 16 kHz/512, with RES enabled and
context-only, with an unchanged duty census and final delay/confidence. The
leak did reach the branch on all four of those streams — the peak trajectory
moves on 4,540 of 5,233 hops (doubletalk, 16 kHz/256), 2,379 of 2,616
(doubletalk, 16 kHz/512), 951 of 1,250 (`aec_record`, 16 kHz/256) and 202 of
625 (`aec_record`, 16 kHz/512) — but it never reaches a DECISION, because the
peak tops out at 3.03 / 2.78 / 1.16 / 0.00 dB and the watchdog does not arm
below 6 dB. The same two captures resampled to 8 kHz and to 48 kHz behave the
same way at 8 kHz/256 and 48 kHz/1024: the peak moves on 2,390 and 3,512 hops
of the doubletalk clip and 228 and 679 of `aec_record`, nothing else in either
trace moves, and output stays byte-identical. Those two grids therefore have
a proven value change and a proven live branch
but no output-moving case in hand: the blind corpus is 16 kHz, so the
1,664-stream sweep above could not cover them.

One synthetic scene in the suite moves visibly and is worth naming, because
it is the shape a real echo path can take. `test_delay_backward_quarantine`'s
multipath scene adds a stronger, earlier path while the old reflection
survives; on the row where the two have EQUAL gain — a near-tie for the
dominant-peak aggregator, so the most scheduling-sensitive row there is — the
re-lock moves from hop 1515 (24.2 s) to hop 2071 (33.1 s). It still re-locks,
still onto the same lock class, and the guarded build still trails the
unguarded one by exactly one window, which is what that block asserts. The
three other gain rows move by at most 5 hops. The test's observation window
grew 1600 -> 2400 hops to outlast the slow row, with the margin now stated in
the source instead of being 1.4%.

No ABI change: `sizeof(Aec)` stays 5,848 B, every field offset is unchanged
(the new float occupies what was tail padding ahead of the census counters),
and `aec_get_mem_size()` is identical at all four grids. No pipeline layout
version moves. Python is untouched: the duty machine is a documented C-only
divergence — the Python orchestrator analyses every hop — so the AIAEC
behavior hash does not move. The 4-channel wrapper in the integration repo
holds its own copy of this machine and follows separately.

`docs/timing_constant_inventory.md` entry `duty-cycle ERLE watchdog peak leak
rate` moves to already-retimed/closed; Retime 32 -> 31, already retimed
14 -> 15. Regenerating that file also repaired drift it was already carrying:
`gen_timing_inventory.py --check` failed before this change, because two
entries (the retired pre-echo bias window and the retired PBFDAF
`alpha_power`) had been annotated by hand in the Markdown with blockquotes
the JSON does not produce. The JSON already recorded both retirements in its
own fields, so the verdicts are unchanged and the prose remains in git
history — but the document is generated, and it is now generated again.

---

## [Unreleased] — 2026-08-26 — the linear filter stops recomputing what it already knows (ABI: `sizeof(Aec)` +16 B, pool +5,664 B at 16 kHz/256)

### Changed

1. **Per-partition `|X|²` is cached, not re-derived.** Both X² partition sums —
   the main filter's Kalman gain and the shadow's NLMS gate — rebuilt
   `Σ_p |X_p[k]|²` from the complex far history on every update call, running
   the scaled-hypot magnitude kernel over every partition of a ring in which
   exactly one row changes per hop, and whose changed row the frontend had
   already squared that same hop. The filter now keeps an `|X_buf|²` mirror
   written as each partition enters history; the sums read it in the same
   partition order, one add per element, so every float is bit-identical while
   a divide and a square root per bin per partition disappear. The
   avg-render-reverb step in the post filter, which squared two more partition
   rows every hop, reads the same mirror.

   The mirror is wiped wherever the history it mirrors is wiped, through a
   single `pbfdaf_clear_far_history()` that owns both arrays — a caller
   clearing `X_buf` alone would leave the sums adding magnitudes of a history
   that is now zero.

2. **The per-bin ERL publish runs only when the taps moved.** `Σ_p |W_p[k]|²`
   was recomputed and republished every hop, over a `W` that changes on about a
   quarter of them (measured 21–26% on two captures) and never during far
   silence. Every writer of `W` now marks the filter, and the publish — the
   flag's only consumer — recomputes on those hops and clears it. The published
   array persists between hops, so the skipped hops serve the floats they would
   have recomputed.

3. **Three magnitude passes that duplicated an existing array are gone.** The
   error energies at step 13 reuse the `|error_spec|²` arrays step 12 and step
   8.5 already filled from the same untouched spectra; the RSA update and the
   stationarity refresh read a far-PSD hold the frontend writes, instead of
   squaring `far_spec` a third and fourth time. The REDUCTIONS are untouched
   and deliberately different — an error energy folds its tail where the
   e2_ref/e2_coa sums do not, and the two trees round differently by design.

   The hold tracks `far_spec`'s lifecycle, not the history ring's: a realign
   clears the ring but not the spectrum, so the two genuinely disagree there
   and the hold is what stays correct.

4. **The far-power EMA is deleted.** `PBFDAF.power` was maintained on every hop
   of both filters — a cold-start branch, two K-wide reductions and a K-wide
   combine — and read by nothing: not the gain, not the noise gate, not the
   orchestrator, not a debug accessor. The array and its grid-retimed retention
   `alpha_power` are removed with it. `test_rate_structural`'s (d4) check,
   which recovered that retention from the filter's own state to prove a
   retimed constant reached the audio path, now recovers `Aec::alpha_pow` —
   the per-sample near/error power EMA behind convergence detection — the same
   way; it still separates 0.9025 / 0.95 / 0.983048 across the four grids, so
   the un-retimed literal still fails it.

   `Aec` also gives up three per-hop scratch arrays: two the far-PSD hold
   retired, and one that had no reader before this change.

### Added

5. **`AecStageTiming.steering_us`** — the fifth stage bucket, covering
   everything between the main filter and the post block: the stationarity
   refresh, the `e2_coarse`/ERL publish and coarse rescue, the double-talk
   analyzer, EPV, `shadow_rise` and the misadjustment estimator. That window
   had no stamps at all, so it read as free while costing 1.15–1.22 µs against
   a four-bucket sum of 10.1–12.5 µs (measured, balanced 16 kHz, two captures x
   two configurations) — every breakdown built on these fields understated the
   hop by that 10–12%. It reuses the two stamps that already closed `linear_us` and
   opened `res_us`, so the count stays seven clock reads per hop and a
   `PROFILE=0` build is unchanged: no clock is linked and all five fields read
   0.

   `make audit-profile` gained the matching value-level assertion. It builds
   the library against a stamp that advances by one per read, so each window's
   value is the number of stamps it spans, and pins the exact tuple
   (1, 2, 1, 1, 1). A stamp that is added, removed or moved changes it; a fast
   or slow machine does not. Without it a missing opening stamp reports 0,
   which is indistinguishable from a stage that cost under a microsecond.

### ABI

`sizeof(Aec)` 5832 -> 5848 B (`AecStageTiming` 16 -> 20 B). The pool moves by
more than that, because both filters carry the new `|X|²` mirror and far-PSD
hold and give back the far-power array, and `Aec` gives back three scratch
arrays:

| Grid | KISS | NE10 | delta |
|---|---:|---:|---:|
| 16 kHz / 256 | 379,776 -> 385,440 | 379,168 -> 384,832 | +5,664 |
| 16 kHz / 512 | 508,848 -> 513,968 | 507,472 -> 512,592 | +5,120 |
| 48 kHz / 1024 | 1,167,072 -> 1,185,536 | 1,164,160 -> 1,182,624 | +18,464 |
| 8 kHz / 256 (legacy) | 275,696 -> 278,256 | 275,088 -> 277,648 | +2,560 |

Measured at `delay_mode = MATCHED`, `delay_num_filters = 5`. The delta is per
INSTANCE, not per delay filter: every `delay_mode` / `n` / `enable_shadow`
combination on a grid moves by that grid's constant, and every difference
between them (5,728 B per delay filter, the `FIXED` byte/ms slope, the mode
gaps) is unchanged — except `enable_shadow = 0`, which carries one filter
instead of two and therefore moves by less (+2,048 B at 16 kHz/256).

Every caller that carves an `Aec` out of its own pool moves with it — in this
stack that is five pipeline layout versions, one per pool: the mono pipeline,
the mono ULCNet variant, the 4ch core (four lanes, so 4x the per-instance
delta), the 4ch wrapper and the 4ch ULCNet wrapper. Carve ORDER changes inside
both filter structs (fields appended at the tail, `power` removed) and inside
`Aec`, so a persisted descriptor is stale for layout reasons as well as size.

`PBFDAF` loses its public `power` and `alpha_power` fields and gains
`x2_cache`, `far_cmag2_hold` and `w_dirty`; `Aec3PostRunIn` gains an optional
`x2_cache` pointer, NULL-safe for a caller driving the post filter without a
PBFDAF behind it.

**Output contract intact**: byte-identical `aec_process()` output and
byte-identical `aec_get_res_context()` seam (`formed_hop` + `error_spec`) on
two captures x two configurations x both FFT backends, per hop, for every step
above.

## [Unreleased] — 2026-08-25 — linear seam rejects unusable over-output

### Fixed

1. The no-RES/no-context C path now advances the same adaptation, ERL and
   double-talk state as Python. RES-enabled production output is unchanged;
   dedicated no-RES C/Python tests cover the formerly skipped branch.
2. In the context-only path used by an external post-filter, when filtering
   quality has not declared the linear estimate usable and the selected
   residual has more hop energy than its microphone capture, the formed
   linear seam selects capture as its third candidate. The internal-RES path
   cannot reach it: the candidate is admitted only under `context_only`
   (`!enable_res && return_res_context`), so with `enable_res` the selection is
   the plain refined/coarse one and the existing frequency-domain guard still
   owns the rule -- byte-identical by construction, in C and in Python, with no
   capture needed to establish it. Running both would let that guard's
   frequency-domain hard switch overwrite the crossfade this selection just
   started. The existing 30-sample selection transition remains active, so this
   is a steady-state fallback rather than a strict bound on every transition
   sample.
3. Python and C now make that selection with the same float32 square and
   pairwise-reduction arithmetic. This decision publishes both the formed hop
   and its matching spectrum; precision disagreement at the threshold is
   therefore an output-contract issue, not merely a diagnostic difference.
   C already reduced in float32, so this moves Python only -- onto C's answer,
   in EVERY configuration including `enable_res=1`. Measured disagreement
   between the old float64 reduction and the new one, replaying the selector
   both ways: 18 of 5043 hops (0.36%) and 3 of 5233 (0.06%) on two AEC
   Challenge double-talk captures under `wav/`, 0 of 1250 on the 10 s pair in
   `wav/aec_record`. Any Python-side 800-case baseline predating this entry is
   stale.

### Changed

4. **`AecConfig` rejects `return_formed_output` without the chain that
   produces it.** The flag now requires `enable_res` or `return_res_context`,
   and the FORM-only branch that ran the selector standalone is deleted. That
   configuration never entered the post chain, so the filtering-quality
   verdict stayed at its constructed `False` for the whole stream and the
   capture fallback above degraded to a bare energy comparison -- a different
   signal from the one the shipped seam produces. A caller asking for the
   formed output alone now raises `ValueError` at construction and must add
   `return_res_context=True`. Python-only: the C config has no such field, and
   `aec_get_res_context()` already leaves the formed seam NULL unless the
   chain ran.

### Impact

All four items change what some caller receives. Items 2 and 3 change
`linear_error` on the context seam; item 3 additionally moves Python's
`enable_res=1` selector decision; item 1 moves the C output of the linear-only
`enable_res=0 && return_res_context=0` configuration, whose filter now steers
from state that used to stop advancing; item 4 turns a previously constructible
Python config into a construction error. The behavior hash must
change, no waveform-equivalence migration is permitted, and AIAEC data must be
rematerialized and repacked before training a release checkpoint. Existing
checkpoints remain useful for an explicit `--input-is-linear-error` diagnostic
against newly materialized WAVs, but their old frontend contract must not be
silently restamped.

---

## [Unreleased] — 2026-08-21 — the product delay is the DOMINANT matched-filter peak (BEHAVIOUR CHANGE: `linear_error` moves)

### Changed

1. **`MatchedFilterLagAggregator.aggregate()` / `da_aggregator_aggregate()`
   report the dominant highest-peak candidate, never the pre-echo one.**

   Both sides qualified the estimate with the *dominant* peak's histogram and
   then reported the *pre-echo* candidate. That is upstream AEC3's behaviour
   and correct there -- its estimator feeds a `RenderDelayController` that
   wants the earliest echo onset. Here the reported sample delay aligns a
   832-tap PBFDKF (52 ms at 16 kHz) directly, so an onset further ahead of the
   dominant echo than the filter spans puts the echo it must cancel outside
   the filter entirely -- while confidence reads 1.0, because confidence
   described a candidate that was not the one returned.

   Measured on a far-end single-talk capture whose true bulk delay is 6210
   samples: the applied delay oscillated between 6144 and 4608 ten times and
   latched on 4608, and 3--13 s ERLE was 1.37 dB. Reporting the dominant peak
   settles at 6212 with two changes and 5.51 dB -- against a measured oracle
   linear ceiling of 6.96--7.03 dB for that recording, i.e. within ~1.5 dB of
   what any linear AEC could do on it.

   The pre-echo histogram is still computed and still reset with the
   aggregator. It is the upstream reference and a diagnostic; only its role in
   selecting the production delay is gone. This is an intentional divergence
   at one seam, not an AEC3 parity fix.

2. **`test_delay_backward_quarantine`'s mis-lock rows are inverted, and two
   are removed.** That quarantine was built to DELAY this mis-lock, never to
   cure it -- its own docstring said curing it was estimator work. With the
   estimator fixed the scene no longer mis-locks, so the row that reproduced
   the defect now guards that it stays fixed, end-to-end through
   `aec_process()`. The rows that measured how long acceptance was delayed,
   and swept the window to prove the expiry released it, are gone with their
   subject; the multipath rows still engage the guard and are still released
   by the expiry at exactly one window, which carries that claim.

   ⚠ The inst-ERLE arm of the predicate no longer has a scene demonstrating
   it is the deciding one -- the pre-echo mis-lock was that scene. It is
   currently unproven rather than shown unnecessary. `delay_backward_quarantine`
   remains default OFF.

### Impact

⚠ **This moves samples in `linear_error`.** It is not frontend-equivalent, so
`aec_behavior_hash` changes and no migration may connect the old waveform to
the new one. AIAEC corpora must have their `linear_error` channel recomputed
and repacked, and checkpoints trained on the old distribution retrained.
No ABI movement: no public struct, allocation size, init signature or enum
changes.

## [Unreleased] — 2026-08-23 — `make PROFILE=1`, and a substitutable stamp for targets without POSIX time

### Added

1. **`PROFILE=0|1`** — the same one-word switch the two consuming pipelines
   already spell that way, so a profile build of the whole chain no longer
   means a hand-written `EXTRA_CFLAGS=-DAEC_STAGE_TIMING=1` here and a
   `PROFILE=1` there. `PROFILE=1` is exactly `-DAEC_STAGE_TIMING=1`; default
   0, unchanged behaviour. Orthogonal to `DEBUG`/`make debug`, which is about
   a `-g` debug build rather than measurement -- profiling a
   build with logging compiled in measures a binary nobody ships.

2. **`AEC_NOW_US`** — the microsecond stamp, substitutable at build time.
   `clock_gettime(CLOCK_MONOTONIC, ...)` is POSIX rather than C99 and this
   library builds for targets with a reduced libc, so

       make PROFILE=1 EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'

   names the platform's own timer -- a plain identifier, because this repo's
   FP policy rejects parentheses in `EXTRA_CFLAGS`. The default is then
   neither compiled nor linked and `<time.h>` is not included. Verified by
   symbol reference across all three configurations: `libaec.a` carries zero
   `clock_gettime` relocations at `PROFILE=0`, one at `PROFILE=1`, and zero
   again at `PROFILE=1` with the stamp substituted.

   The substitute must be monotonic: a stamp that can go backwards makes the
   unsigned subtraction produce a nonsense interval near the full 32-bit
   range rather than a small wrong one. Naming a function that returns a
   constant is a supported way to keep the flag on and pay nothing.

### Fixed

3. **`PROFILE=1` no longer loses to a command-line `EXTRA_CFLAGS`, and no
   longer arrives twice.** The knob is set through `PROFILE_CFLAGS` folded
   into `CFLAGS`, mirroring `SIMD` two lines below it, rather than appended to
   `EXTRA_CFLAGS`.

   `EXTRA_CFLAGS` was the wrong carrier twice over. A command-line assignment
   beats a plain `+=`, so `make PROFILE=1 EXTRA_CFLAGS=-DFOO` silently
   produced the release configuration -- the same objects, the same CFG_SIG
   directory as `PROFILE=0`, and a report of all zeros with nothing saying
   why. And because `EXTRA_CFLAGS` is forwarded verbatim to every producer
   sub-make, a consumer that also has `PROFILE` delivered the same `-D` a
   second time, splitting byte-identical objects across two CFG_SIG keys and
   re-keying audio_common and NR, neither of which references the macro.
   `CFLAGS` has neither problem: a command-line assignment to it is already a
   hard error, and it is not forwarded.

   The two pipelines now forward `PROFILE` by name, so this library's compile
   receives the flag from its own knob rather than from a `-D` a consumer
   hand-wrote. They still define `AEC_STAGE_TIMING` for their OWN translation
   units, which read it to decide whether to copy this record out -- both
   sides deriving from one `PROFILE` is what keeps the two views from
   disagreeing. A pin older than this commit has no knob to receive and
   degrades to the documented one-half state: the wrapper's own stages
   measure and the AEC block reads zero.

4. **`audit-profile`** — a regression target for a failure mode that is silent
   by construction, in the shape of `audit-no-stdio`. It asserts, in both
   directions, that `PROFILE=0`/`PROFILE=1`/stamp-substituted resolve to three
   distinct keys, that `PROFILE=0` links no clock, that `PROFILE=1` links one
   (without which every other assertion here would pass vacuously), that
   substituting the stamp removes it again, and that a command-line
   `EXTRA_CFLAGS` does not defeat the knob. Each assertion was mutation-tested
   to confirm it fails when the property it covers is broken.

   No ABI or behaviour change: `sizeof(Aec)`, every offset and every sample
   are unchanged in all configurations.

## [Unreleased] — 2026-08-21 — the product delay is the DOMINANT matched-filter peak (BEHAVIOUR CHANGE: `linear_error` moves)

### Changed

1. **`MatchedFilterLagAggregator.aggregate()` / `da_aggregator_aggregate()`
   report the dominant highest-peak candidate, never the pre-echo one.**

   Both sides qualified the estimate with the *dominant* peak's histogram and
   then reported the *pre-echo* candidate. That is upstream AEC3's behaviour
   and correct there -- its estimator feeds a `RenderDelayController` that
   wants the earliest echo onset. Here the reported sample delay aligns a
   832-tap PBFDKF (52 ms at 16 kHz) directly, so an onset further ahead of the
   dominant echo than the filter spans puts the echo it must cancel outside
   the filter entirely -- while confidence reads 1.0, because confidence
   described a candidate that was not the one returned.

   Measured on a far-end single-talk capture whose true bulk delay is 6210
   samples: the applied delay oscillated between 6144 and 4608 ten times and
   latched on 4608, and 3--13 s ERLE was 1.37 dB. Reporting the dominant peak
   settles at 6212 with two changes and 5.51 dB -- against a measured oracle
   linear ceiling of 6.96--7.03 dB for that recording, i.e. within ~1.5 dB of
   what any linear AEC could do on it.

   The pre-echo histogram is still computed and still reset with the
   aggregator. It is the upstream reference and a diagnostic; only its role in
   selecting the production delay is gone. This is an intentional divergence
   at one seam, not an AEC3 parity fix.

2. **`test_delay_backward_quarantine`'s mis-lock rows are inverted, and two
   are removed.** That quarantine was built to DELAY this mis-lock, never to
   cure it -- its own docstring said curing it was estimator work. With the
   estimator fixed the scene no longer mis-locks, so the row that reproduced
   the defect now guards that it stays fixed, end-to-end through
   `aec_process()`. The rows that measured how long acceptance was delayed,
   and swept the window to prove the expiry released it, are gone with their
   subject; the multipath rows still engage the guard and are still released
   by the expiry at exactly one window, which carries that claim.

   ⚠ The inst-ERLE arm of the predicate no longer has a scene demonstrating
   it is the deciding one -- the pre-echo mis-lock was that scene. It is
   currently unproven rather than shown unnecessary. `delay_backward_quarantine`
   remains default OFF.

### Impact

⚠ **This moves samples in `linear_error`.** It is not frontend-equivalent, so
`aec_behavior_hash` changes and no migration may connect the old waveform to
the new one. AIAEC corpora must have their `linear_error` channel recomputed
and repacked, and checkpoints trained on the old distribution retrained.
No ABI movement: no public struct, allocation size, init signature or enum
changes.

## [Unreleased] — 2026-08-23 — `make PROFILE=1`, and a substitutable stamp for targets without POSIX time

### Added

1. **`PROFILE=0|1`** — the same one-word switch the two consuming pipelines
   already spell that way, so a profile build of the whole chain no longer
   means a hand-written `EXTRA_CFLAGS=-DAEC_STAGE_TIMING=1` here and a
   `PROFILE=1` there. `PROFILE=1` is exactly `-DAEC_STAGE_TIMING=1`; default
   0, unchanged behaviour. Orthogonal to `DEBUG`/`make debug`, which is about
   a `-g` debug build rather than measurement -- profiling a
   build with logging compiled in measures a binary nobody ships.

2. **`AEC_NOW_US`** — the microsecond stamp, substitutable at build time.
   `clock_gettime(CLOCK_MONOTONIC, ...)` is POSIX rather than C99 and this
   library builds for targets with a reduced libc, so

       make PROFILE=1 EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'

   names the platform's own timer -- a plain identifier, because this repo's
   FP policy rejects parentheses in `EXTRA_CFLAGS`. The default is then
   neither compiled nor linked and `<time.h>` is not included. Verified by
   symbol reference across all three configurations: `libaec.a` carries zero
   `clock_gettime` relocations at `PROFILE=0`, one at `PROFILE=1`, and zero
   again at `PROFILE=1` with the stamp substituted.

   The substitute must be monotonic: a stamp that can go backwards makes the
   unsigned subtraction produce a nonsense interval near the full 32-bit
   range rather than a small wrong one. Naming a function that returns a
   constant is a supported way to keep the flag on and pay nothing.

### Fixed

3. **`PROFILE=1` no longer loses to a command-line `EXTRA_CFLAGS`.** A
   command-line assignment beats a plain `+=` in a Makefile, so
   `make PROFILE=1 EXTRA_CFLAGS=-DFOO` built the release configuration --
   same objects, the same CFG_SIG directory as `PROFILE=0`, and a report of
   all zeros with nothing anywhere saying why. The append is now `override`.
   Verified by build-directory hash: the two invocations now resolve to the
   same profile directory and `PROFILE=0` to a different one.

   No ABI or behaviour change: `sizeof(Aec)`, every offset and every sample
   are unchanged in all three configurations.

## [Unreleased] — 2026-08-21 — `AEC_STAGE_TIMING`: the stage timing compiles out, and is OFF by default

### Added

1. **`-DAEC_STAGE_TIMING=0`** — compiles every per-hop stage stamp out.
   `aec_get_last_timing()` and `AecStageTiming` stay, and all three fields then
   read 0 on every hop, which is how a caller distinguishes a build that does
   not measure from one that does.

   Measuring costs **five `clock_gettime()` calls per hop per instance** —
   twenty in the four-lane pipeline before its own — which is real on a target
   where the clock is not a cheap vDSO read. Verified by symbol reference:
   `aec.o` carries one `clock_gettime` relocation by default and none under
   `AEC_STAGE_TIMING=0`.

   **Default OFF.** A diagnostic that has to be switched off is one that ships
   on by accident, so a release build carries none of it and a profile build
   asks for it: `-DAEC_STAGE_TIMING=1`, or `make PROFILE=1` in the pipelines
   repo, which also turns on that repo's own wrapper-side half.

   ⚠ This flag decides what the LIBRARY records. A consumer with its own
   display-side profiling flag must be built against a library that has this
   one, or its report renders zeros.

   No ABI change: `sizeof(Aec)` and every offset are unchanged, and
   `test-static-aec` reports all 669,824 samples byte-equal either way.

## [Unreleased] — 2026-08-20 — `aec_get_last_timing()`: per-hop cost of the three largest stages (ABI: `sizeof(Aec)` +16 B)

### Added

1. **`aec_get_last_timing(const Aec*, AecStageTiming*)`** — the per-hop
   wall-clock cost of `frontend`, `linear` and `res`, in microseconds. A
   pipeline could already time a whole `aec_process*()` call from outside, but
   the three stages inside it were opaque, and on the 4-lane spatial pipeline
   that opaque block is the largest single item in the hop. Diagnostic only:
   nothing in the engine reads the record back and no processing depends on it.

   - `frontend_us` — entry through the shadow filter: mic HPF, saturation,
     delay estimation and ring alignment, render activity, `mu_scale`,
     mic-clip, RSA update. Cheap under `AEC_DELAY_EXTERNAL_ALIGNED`, where the
     estimator does not run.
   - `linear_us` — the main adaptive filter, including the far-end FFT on an
     instance that computes its own rather than borrowing one through
     `aec_process_context_shared_far()`.
   - `res_us` — the AEC3 post/RES block. It runs when EITHER `enable_res` or
     `return_res_context` is set, so a context-only caller pays it and sees a
     real number here rather than a zero.

   The three deliberately do not sum to the call. The remainder — stationarity
   refresh, `e2_coarse`/ERL publish, the DT analyzer, EPV, `shadow_rise`, the
   misadjustment estimator, the power EMAs and convergence detection — is left
   for the caller to carry explicitly, because attributing it would mean
   inventing boundaries the code does not have.

   Cleared at the top of every `aec_process*()` call, so a hop reports only
   the stages it actually reached. Microsecond resolution: a stage under 1 us
   reads 0.

### ABI

`sizeof(Aec)` 5816 -> 5832 B; `ALIGN16(sizeof(Aec))` 5824 -> 5840, so
`aec_get_mem_size()` grows **+16 B per instance** (measured: 379152 -> 379168
at 16 kHz balanced). Every caller that carves an `Aec` out of its own pool
moves with it — in this stack that is five pipeline layout versions, one per
pool: the mono pipeline, the mono ULCNet variant, the 4ch core (four lanes, so
+64 B), the 4ch wrapper and the 4ch ULCNet wrapper. Carve order and buffer set
are unchanged, so no `build_flags_hash` moves and each downstream
`layout_version` counter is the only signal that a persisted descriptor is
stale.

The engine now calls `clock_gettime(CLOCK_MONOTONIC)`, which it previously
never did — four reads per hop. This is a new libc-time dependency for
integrators who were linking against a time-free library.

**Output contract intact**: the timing writes touch no signal path; all
15 `test-*` targets and `selftest` pass unchanged.

## [Unreleased] — 2026-08-19 — runtime strength retarget: `aec_set_preset()` + a dB-domain ramp (ABI: `sizeof(Aec)` +16 B)

### Added

1. **`aec_set_preset(Aec*, AecPreset, float ramp_ms)`** — change the residual-echo
   strength on a RUNNING instance. The three shipped presets differ in exactly one
   field (`min_gain_floor_far_active_db`), and that field reaches the suppressor as
   a single scalar clamp read once per hop, so a preset change is a retarget rather
   than a rebuild: the filter, the delay lock, every smoothing history, the
   far-active latch and the DominantNearend counters all carry on. Unlike
   `aec_config_from_preset()`, which falls back to balanced for an unknown value,
   this REFUSES one — a setter has a caller to tell.

2. **`suppression_gain_set_split_floor_far_active_db(SuppressionGain*, float db,
   float ramp_ms)`** — the primitive underneath it, public because the 4-channel
   product's shared post-stage suppressor is a `SuppressionGain` that no `Aec`
   owns. `ramp_ms == 0` applies on the next hop and lands on exactly the float a
   fresh instance built with that dB would hold; a positive `ramp_ms` walks there
   linearly in dB (geometrically in the power domain, so no logarithm per hop),
   bounded at 60 s. `db` is validated against the config validator's own
   `[-300, 50]`, so a runtime call cannot install a value init would have refused,
   and every argument is checked before the first store — a refusal leaves the
   instance bit-identical.

   Note for 4-channel integrators: a lane built with `spatial_linear_context`
   never reaches `suppression_gain_get_gain()`, so retargeting a lane is inert by
   construction. Retarget the shared post-stage suppressor instead.

### Changed

3. **ABI: `sizeof(SuppressionGain)` 384 → 392 B, `sizeof(Aec)` 5,808 → 5,816 B,
   and `ALIGN16(sizeof(Aec))` 5,808 → 5,824 B — a uniform +16 B on every pool
   total.** Two floats are appended to `SuppressionGain`'s STATE block (the live
   floor and its per-hop ramp factor); `SuppressionGainConfig`'s layout is
   unchanged. The growth crosses a 16-byte boundary, so unlike a change that
   `ALIGN16` absorbs, this one does move the carve base and every offset after it.
   No per-bin array is added: the SuppressionGain sizing terms and carve are
   untouched, and every DELTA is unaffected (the 5,728 B/filter step, the `FIXED`
   ring formula, and every mode-to-mode difference — both sides of a subtraction
   carry the same +16 and it cancels).

   Measured totals, balanced, all four grids:

   | grid | KISS n=5 | KISS n=1 | NE10 n=5 | NE10 n=1 |
   |---|---:|---:|---:|---:|
   | 8 kHz/256 (legacy) | 275,680 | 252,768 | 275,072 | 252,160 |
   | 16 kHz/256 (default) | 379,760 | 356,848 | 379,152 | 356,240 |
   | 16 kHz/512 | 508,832 | 485,920 | 507,456 | 484,544 |
   | 48 kHz/1024 | 1,167,056 | 1,144,144 | 1,164,144 | 1,141,232 |

   `EXTERNAL_ALIGNED` (KISS): 176,208 / 214,752 / 343,824 / 738,528 on the same
   four grids.

   **A caller that hardcoded a byte count must re-query `aec_get_mem_size()`.**
   Every downstream pipeline descriptor bumped its `layout_version` for this.

4. **`suppression_gain_get_gain()` reads the live floor rather than
   `cfg.split_floor_far_active` directly.** The config field is now the ramp's
   TARGET. They are seeded equal and stay equal on any instance whose setter is
   never called, so the arithmetic is unchanged there — verified byte-equal over
   3,600 hops on three product grids against the previous build.

5. **`aec_config_from_preset()` now resolves its strength table through a shared
   helper** so the factory's documented balanced fallback and the setter's refusal
   read from one table. No behavioural change.

### Python

6. `AEC.set_preset(preset, ramp_ms=0.0)` and the suppressor primitive mirror the C.
   The ms→hops conversion is deliberately computed in float32 in the C helper's
   operation order: both round half to even, but a double disagrees with the
   shipping C wherever the value lands near a half hop (at 16 kHz/hop 128,
   `ramp_ms = 20` is exactly 2.5 hops in double and 2.5000002 in float). The
   behaviour hash moves; the frozen dataset frontend never reaches the changed
   code (`get_gain` is not called on that path) and renders byte-identical.

## [Unreleased] — 2026-08-18 — Python: `AEC.apply_external_realign()`, the caller-side far realign that is not a reset (behaviour-hash bump)

### Added

`AEC.apply_external_realign(delta_samples)` gives an `EXTERNAL_ALIGNED`
instance a response to its caller changing the far alignment other than
`reset()`. It is the Python specification of the C
`aec_apply_external_realign()` shipped in `05dbcde` / `68549b8`; the reference
had none, so `pipelines/4ch_aec_bf_nr_res/pipeline.py` was still resetting all
four lanes on every admitted shared-delay change — the behaviour the C side
abandoned, and the spectrogram "vertical line" it produces (dozens of hops of
re-exposed echo plus one near-zero output hop).

* Evidence gate: `linear_is_cancelling()` — both readings, not the inst-ERLE
  ring alone, which ages out across a stretch of far silence — AND `|delta|`
  inside the tap span.
* Warm outcome (returns 1): `PBFDAF.warm_shift_ir()` now takes a SIGNED shift.
  Positive = the aligned far advanced, the IR moves toward tap 0; negative =
  it retarded, the IR walks the other way and the per-partition far history
  (`X_buf`) is cleared, because a held history that is AHEAD of the new stream
  mis-estimates the echo against the shifted taps. The retard therefore costs
  a bounded refill transient, `ceil((|delta| + the response's tap offset) /
  hop_size) + 1` hops, whose residual is not bounded by the echo either.
* Reject outcome (returns 0): a FILTER-ONLY restart,
  `_reset_filter_for_realign()`. Taps, per-partition far spectra, far-power
  normalizer, the far half of the analysis frame, and the convergence / mu /
  stationarity latches go; the AEC3 post chain (i.e. the synthesis overlap-add)
  and the MIC half of the analysis frame stay. Unshifted taps sit at the
  alignment the caller abandoned and subtract a decorrelated copy of the echo,
  so keeping them is worse than having no filter; wiping them without the
  synthesis restart is the whole reason the entry point exists.
* `-1` outside `EXTERNAL_ALIGNED`; `delta_samples == 0` is a no-op returning 0.

Refactors that support it and change nothing on their own: `PBFDAF.reset()` is
now `reset_taps()` (adaptation state) plus the two time-domain analysis
buffers, and `_reset_filter_derived_state()` is `_reset_filter_latches()` plus
the AEC3 post chain and the pending-delay clear — both halves shared verbatim
with the realign path, so the two restarts cannot disagree about what a
restarted filter may remember. `PBFDAF.tap_span` and the existing
`get_time_domain_filter()` become the single spelling of the filter's reach and
of its concatenated IR, replacing the three and two open-coded copies the two
warm-transfer sites had between them.

`python/tests/test_external_realign.py` carries six tests over the C test's
semantics, with a `reset()` twin as the can-fail control on both the warm path
(the defect reproduces) and the reject path (the trajectory to match, and the
silent output hop to avoid). Reverting each mechanism turns its own row red.

### ⚠ Contract-version event: the AIAEC behaviour hash moves

The note carried since the 2026-08-13 entry — that `python/` is left untouched
because the AIAEC dataset contract pins an AST hash of the mirrored
`lib/aec/python` tree, and any edit there invalidates existing shards and
checkpoints — is cashed in here. The hash
(`AIAEC/dataset_gen/aec_behavior_hash.py`, schema `canon-ast-1`) moves with ANY
change under `python/modules/`:

```
8198bd0e29e4530f16a1ada6eafb86148e09ec749d10e84771bf2c86598143cd
  -> 7c9249dee507f31837e6025c91ca855c05c59febcdee1cad5189cba7ce49cf88
```

No shard needs rematerializing and no checkpoint is invalidated: the pair is
recorded in `ACCEPTED_BEHAVIOR_HASH_MIGRATIONS`
(Audio_ALG `AIAEC/dataset_gen/linear_aec.py`) with the byte-identity evidence
the table requires — `linear_error` and `echo_hat` render byte-for-byte
unchanged (2,048,000 bytes per stem) on the frozen frontend's 32 s,
16 kHz/512/256, 400 ms-delay scene, which reaches both refactored reset helpers
on that render, and `warm_shift_ir`'s positive-shift output is pinned
byte-identical to the pre-change body across 30 (grid, shift) pairs on three
grids. The migration entry goes live when Audio_ALG's `lib/aec` pin moves to
this commit; the two are one coordinated event.

`__version__` is deliberately still `4.0.0rc1` — it is stripped from the
behaviour hash by design, and the fold-in stays scheduled for the next
coordinated contract-breaking release.

### Notes

- `aec_process()` output is unchanged for any caller that never invokes the
  new entry point: the two splits re-group the same statements in the same
  order (one independent pair transposed — the AEC3 post chain now runs after
  the warmup re-arm), and `warm_shift_ir`'s added arm is the `s < 0` one, which
  the single existing caller's `0 < new_delay < reach` gate cannot reach.
- `pytest python/tests`: 243 passed (237 before, +6 new rows).
- Python-only step. `c_impl/` is untouched here; the C side already carries
  this mechanism.

---

## [Unreleased] — 2026-08-17 — `delay_backward_quarantine_*`: bounded backward-jump quarantine on a delay CHANGE (default OFF)

### The defect

With a white-noise far end at a true bulk delay of 6400 samples (16 kHz,
fft 512 / hop 256, `MATCHED` n=5) the matched filter acquires CORRECTLY at
6336 on hop 32, then on hop 49 re-locks to 4800 — exactly 1600 samples
(100 ms) EARLY, the pre-echo mis-attribution signature — and keeps reporting
4800 for the rest of the run. The wrong candidate is SUSTAINED, so no
K-consecutive-confirmation gate can reject it: every confirmation window it
is offered agrees with itself. `pipelines/4ch_aec_bf_nr_res` reproduces the
same signature through its own shared estimator (hop 34 → 6336, hop 50 →
4800).

This is the inverse of the out-of-range mis-lock already documented in
`docs/delay_estimator_design_zh_TW.md` §2.2: here the true delay is well
inside the bank's reach and the first answer is right.

### The mechanism

Three properties, and the mechanism is unsound without all three.

**Directional.** Only a candidate strictly EARLIER than the delay in force
is quarantined. Pre-echo mis-attribution cannot produce a LARGER answer, so
a forward jump is not this mechanism's business and is never held. First
acquisition is not either — that stays `delay_acquire_protect_converged`.

**Evidence-gated.** The candidate is held only while the linear filter is
still demonstrably cancelling at the applied alignment.
`aec_linear_is_cancelling()` (new, public, read-only) is that test, and is
the single definition both this engine's Path-B arm and the 4ch wrapper's own
arm consult. It reads two figures the engine already maintains, both against
thresholds Path A's acquire-time guard already uses: last hop's windowed ERLE
against 2.5 dB, and the recent inst-ERLE peak against
`cfg.delay_acquire_inst_erle_db`. Both arms are load-bearing:

- windowed-only cannot see this defect. The re-lock lands 17 hops after the
  first acquisition, inside the lag `delay_acquire_inst_erle_db`'s own field
  comment documents; measured at the acceptance hop, windowed ERLE is
  1.660 dB (Python) / 1.804 dB (C) against its 2.5 dB threshold, while the
  inst peak is 7.269 dB. Pinned in C by configuring the fast arm out of reach
  and showing the quarantine stop engaging.
- inst-only would release between far-active bursts, where the ~15-frame ring
  ages out while the filter is still cancelling. Pinned in Python by the
  backward-move scene: at the unguarded acceptance hop the inst peak is
  −0.934 dB and windowed ERLE is 3.728 dB, so the slow arm alone engages.

**Time-bounded for a continuously qualifying backward candidate.** The
countdown is armed on the first refusal and spends one tick per estimation
cycle while such candidates continue to be offered; at expiry the candidate
is ACCEPTED. Candidate values may move within that qualifying class without
re-arming the countdown. Cancellation collapse, a forward/non-qualifying
estimate, or a confidence gap disarms it immediately; a later backward episode
starts a new window. This is persistence gating, not a wall-clock deadline
spanning unrelated estimator episodes.

There is no dominance early-release, and that is a finding rather than an
omission: `delay_aec3_confidence()` is histogram consistency quantised to
0 / 0.5 / 1.0, carries no match-quality information (the pre-echo candidate
reaches REFINED too), and Path B already gates on it. Persistence is the only
signal available, and the window is what measures it.

**A mis-lock is therefore DELAYED by one window, never cured.** Curing
pre-echo mis-attribution is estimator work and is out of scope here.

### Measured

Pre-echo scene, acceptance hop of the WRONG delay (16 kHz / fft 512 /
hop 256):

| window | = cycles | lib/aec | 4ch core |
|---|---|---|---|
| off | — | 49 | 50 |
| 0.5 s | 31 | 80 | 81 |
| 1.0 s (default) | 62 | 111 | 112 |
| 2.0 s | 125 | 174 | 175 |
| 4.0 s | 250 | 299 | — |

In this sustained pre-echo scene, acceptance lands on unguarded_hop +
window_cycles exactly on every implementation: the predicate remains active,
so expiry is the release path. Other scenes may release earlier when
cancellation evidence disappears.

Multipath / path addition (C, old path at 2400 surviving at g_old, stronger
new path at 1600 / gain 0.5 appearing at hop 250, 1600 hops) — the family the
previous predicate vetoed forever:

| g_old | re-lock off | re-lock on | cost |
|---|---|---|---|
| 0.2 | 498 | 498 | 0 |
| 0.3 | 488 | 488 | 0 |
| 0.4 | 495 | 557 | +62 |
| 0.5 | 1515 | 1577 | +62 |

All four rows re-lock, worst case exactly one window; the two zero-cost rows
are the early release firing. The Python reference on the same geometry
(620 hops): 310/323/394/510 → 310/385/456/572, same shape.

A forward move (64 → 3000 at hop 375) has a byte-identical accepted-delay
trajectory with the mechanism on and off, in all three implementations.

### Why the default window is 1.0 s

Lower bound from the defect's dynamics: the pre-echo re-lock arrives 17 hops
after the first lock, so the window must be comfortably longer than that to
cover the interval in which a mis-lock is most damaging — 62 cycles is 3.6×.
Upper bound from the cost of a real change: in the multipath table the
estimator itself needs 488–1515 hops to produce the new candidate, so 62 more
is 4–13% of a latency the guard did not create, and it is only ever paid when
cancellation has NOT collapsed.

### Behaviour and cost

`AecConfig::delay_backward_quarantine_enabled` (default 0) gates Path B only;
`delay_backward_quarantine_s` (default 1.0, seconds, converted to cycles once
in `aec_carve()` against the resolved grid with a floor of 1) is inert while
the enable is 0. Both ERLE readings sit at 0 until the machinery that fills
them has run, so a configuration that never fills them leaves the mechanism
inert (fail-open); it never blocks on an absent metric.

Default OFF keeps the shipped path byte-exact, so no 800-case gate is
required for this entry. Verified rather than assumed: the NN-seam
configuration (`enable_res=0`, `return_res_context=1`) rendered with the
mechanism off is byte-identical over 512,000 bytes to the pre-change build.

### ABI

`sizeof(AecConfig)` 196 → 204 B (one `int` + one `float`), and `Aec` gains the
quarantine countdown pair, so every `aec_get_mem_size()` figure shifts by
exactly +16 B — same offset on every mode / bank size / grid, every delta
(5,728 B per filter, the `FIXED` byte-per-ms formula, the mode-to-mode
differences) unchanged — **rebuild required**, no source change for callers
that construct configs
through `aec_config_defaults()` / `aec_config_from_preset()`. `aec.h` gains
one function declaration (`aec_linear_is_cancelling`).

### Tests

`make test-delay-backward-quarantine` (new target, 31 checks) and
`python/tests/test_delay_backward_quarantine.py` (8 tests). Both reproduce
the defect with the mechanism OFF first and pin that as the documentation of
it, then assert that the mechanism only delays it, that the multipath family
re-locks in all four gain rows, and that a forward candidate is untouched.
The 4ch C/Python per-hop parity test (7 tests) sweeps the window on both
implementations and adds a proxy-noise scene, the only one on which
"judge the estimator's own lane" and "judge any lane" are distinguishable.

Mutation matrices are recorded in the three test headers: 9 mutations in C
(1/8/8/4/0/5/1/1/13 rows red), 8 in Python (1/2/1/1/1/1/1/5 tests red) and 8
across the 4ch core + its Python mirror (1 test red each), control green
before and after every one. The single C zero is the windowed-ERLE arm, which
no scene reachable through the public API makes the deciding one on a
duty-cycled analysis; it is red in the Python mirror and the C test header
says so rather than claiming coverage it does not have. Two scenes exist
purely because their absence made a mutation read green: the FORWARD
MULTIPATH scene (the plain forward move cannot see the direction test) and
the 4ch proxy-noise scene (identical per-channel audio cannot see the
proxy-lane rule).

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
  Python also reads it in the P3f filter-state classifier
  (`dt_from_shadow > 0.5` → `_ne_evidence`), but that classifier only writes
  diagnostics; the live AEC3 `usable_linear_estimate()` and RES gain do not
  consume its `refined_usable` result. There is therefore no C/Python audio
  path gap at this point.

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
  **byte-identical to `aec_process()`**. Verified by c_impl/test/stream_sim.c:
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

**Headline**: 1 production change shipped — **Sprint P1 AEC3 FilterAnalyzer port** (single-channel verbatim port of `docs/aec3_extracts/src/aec3/filter_analyzer.cc`, `~250 LOC`, owned by `AecState`; default-True). The port produces a non-zero direct-path delay scalar that AEC3's reverb-tail update consumes — indirectly closes v3.21.5 Sprint C's reverb-tail blocker. Cumulative 800-case bench Pareto-positive vs v3.21.5: FS_static **+0.059** / FS_movement **+0.036** / DT buckets within ±0.01 / NE flat.

Sprints P2 / P4 closed as **intentionally-incompatible with our PBFDKF architecture** — both AEC3 parity items are permanently retired (TransparentMode + AEC3-default-off stationarity); any v3.22+ revisit must be labelled as PBFDKF-specific divergence, NOT AEC3 parity restoration. Sprint P3 ships byte-equal structural parity (canonical control surface for `use_stationarity_properties` now lives at `SuppressorConfig.echo_audibility`, with top-level `aec3_post_stationarity_zero_enabled` retained as deprecated alias).

Cycle close: `docs/v3_21_6_cycle_close.md`. v3.22 entry gate (AEC3 parity baseline locked): **MET** — every Bucket-1 item has a closed verdict.

### Sprint P1 — FilterAnalyzer port SHIPPED (Pareto-positive)

AEC3 `filter_analyzer.cc` produces `FilterDelaysBlocks()` (per-channel direct-path delay) + `ConsistentFilterDetector` (peak-stability gate for `any_filter_consistent`). Pre-v3.21.6, our [`python/modules/state/filter_delay.py:57-60`](python/modules/state/filter_delay.py#L57) received `analyzer_filter_delay_estimates_blocks=None` always (FilterAnalyzer was a v3.18 audit-only stub at `python/modules/filter_analyzer.py`, since deleted) → `min_direct_path_filter_delay()` returned 0 → reverb-tail update never fired on cohort-tail cases. This was the root cause v3.21.5 Sprint C diagnosed but couldn't fix in the v3.21.5 narrow scope.

P1 ships a full single-channel port (block units translated AEC3 `kBlockSize=64` 4ms → our `HOP_SAMPLES=160` 10ms; convergence hold 5s = 500 hops; consistency hold 1.5s = 150 hops). The new `state/filter_analyzer.py` (~250 LOC) covers `ConsistentFilterDetector` + 3-tap 600Hz HPF + region-sweep peak finder + state machine, verbatim against the AEC3 source. `AecState` owns the analyzer; `PBFDAF.get_time_domain_filter()` (new ~10 LOC IFFT concat helper) feeds the time-domain impulse response per hop. Reverb-update `_delay_blocks` switches from legacy `_current_delay // hop_size` to `aec_state.min_direct_path_filter_delay()`. The v3.18 Phase C.A audit-only stub (incompatible API) is deleted.

Verified default-OFF byte-equal preserved (25/25 PASS vs v3.21.5 anchor) before flipping default True. 800-case bench Pareto-positive on FS without DT damage. Verdict: `docs/v3_21_6_p1_filter_analyzer_verdict.md`.

Known limitation (not blocking ship): `fa_consistent=0%` on the LN18k5r8 cohort case — PBFDKF's Kalman peak position is noisier than AEC3's NLMS-stable envelope, so the 1.5s peak-stability detector rarely fires. Effect: `UpdateFilterGain` falls back to running-max path; `TransparentMode.any_filter_consistent` stays False (irrelevant — TM disabled by P2). Does not affect `filter_delays_blocks()` output (P1's primary deliverable). Documented as the PBFDKF-vs-AEC3 architectural-incompatibility note that motivated P2's parity closure.

### Sprint P2 — TransparentMode audit CLOSED intentionally-incompatible

4 mismatch findings vs AEC3 source (`transparent_mode.cc` Legacy variant + `aec_state.cc:189-325` Update flow):

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

AEC3 `echo_remover.cc:495-501`
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

AEC3 `echo_audibility.h:40-51`
+ `residual_echo_estimator.cc:303-313`
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
(dominant_nearend_detector.cc +
residual_echo_estimator.cc:340-358)
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
