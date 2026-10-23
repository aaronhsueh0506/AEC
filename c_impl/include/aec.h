/* Public C API for the single-channel PBFDKF AEC.
 *
 * Python is the algorithm specification; this header exposes the float32 C
 * implementation, heap and caller-owned-pool construction, lockstep and
 * split render/capture processing, and the external RES context seam.
 */
#ifndef AEC_AEC_H
#define AEC_AEC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "fft_wrapper.h"
#include "hpf.h"
#include "saturation.h"
#include "delay_aec.h"
#include "detectors.h"
#include "pbfdkf.h"
#include "epc_shadow.h"
#include "render_signal_analyzer.h"
#include "aec_post.h"   /* AecPost + AecPostRunObj/In/Scratch + sub-modules */

/* Stable public values. The preset factory resolves these to
 * min_gain_floor_far_active_db = -20, -28 and -38 dB. NO_FLOOR mirrors Python
 * AecPreset.NO_FLOOR: it sets min_gain_split_floor_enabled=0 instead of
 * varying the far-active dB value, and leaves min_gain_floor_far_active_db /
 * _dt_db at balanced's -28/-20, which are never read once the gate is off
 * (see aec_config_from_preset() and the `if (cfg->split_floor_enabled)` sites
 * in suppression_gain.c). */
typedef enum {
    AEC_PRESET_MILD       = 0,
    AEC_PRESET_BALANCED   = 1,
    AEC_PRESET_AGGRESSIVE = 2,
    AEC_PRESET_NO_FLOOR   = 3
} AecPreset;

/* ── Delay mode (init-time, immutable for the instance's lifetime) ────────
 * How the far-end is aligned to the mic; the single source of truth. Mirrors
 * Python modules/enums.py AecDelayMode (identical integer values).
 *
 *   AEC_DELAY_MATCHED           the matched-filter bank estimates the delay
 *                               online and a reference ring applies it.
 *                               Requires delay_num_filters in [1, 5] and
 *                               fixed_delay_samples < 0.
 *   AEC_DELAY_FIXED             no estimator; the caller MEASURED the system
 *                               delay (bring-up) and the reference ring
 *                               applies exactly fixed_delay_samples.
 *                               Requires fixed_delay_samples >= 0.
 *   AEC_DELAY_EXTERNAL_ALIGNED  no estimator and no ring; the caller
 *                               guarantees the `ref` it passes is ALREADY
 *                               aligned to `mic`. Requires
 *                               fixed_delay_samples < 0.
 *
 * Illegal mode/field combinations are REJECTED (aec_validate_config), never
 * silently normalised: a field the selected mode cannot honour is a caller
 * mistake. aec_config_resolve_delay() holds the deprecated-field
 * translation. */
typedef enum AecDelayMode {
    AEC_DELAY_MATCHED = 0,
    AEC_DELAY_FIXED = 1,
    AEC_DELAY_EXTERNAL_ALIGNED = 2
} AecDelayMode;

/* AecConfig.spatial_linear_context levels. 0: full context. 1: this lane's own
 * G_res is not computed. 2: linear half of the context only. */
#define AEC_CONTEXT_NO_GAIN     1
#define AEC_CONTEXT_LINEAR_ONLY 2

/* Bad-relock judgment window (Python _DELAY_SHIFT_JUDGMENT_HOPS). Not a
 * config field on either side: 100 is the upper end of the "~50-100 frames of
 * poor cancellation while taps re-converge", not a value to retune per
 * deployment. */
#define AEC_DELAY_SHIFT_JUDGMENT_HOPS 100
/* Fixed cap for Aec::delay_reject_values (see its field doc). */
#define AEC_DELAY_REJECT_CAP 8

typedef struct AecConfig {
    int    sample_rate;        /* 16000 */
    int    fft_size;           /* frame size; 256/512 @16k, 1024 @48k */
    int    filter_length;      /* 832 samples (52 ms) */
    int    n_partitions;       /* derived if 0 */
    float  mu;                 /* 0.3 */
    float  delta;              /* 1e-8 */

    /* toggles */
    int    enable_cng;             /* 1 */
    /* CNG N2 tracker coordinate fix (config.py cng_n2_windowed_psd; mechanism
     * in aec_post.h). Default OFF (0): validated only in an isolated harness,
     * not yet through the 800-case corpus gate required for default-on. */
    int    cng_n2_windowed_psd;    /* 0 */
    /* DEPRECATED translation-layer mirror of `delay_mode` (below), kept so
     * older callers keep compiling and behaving identically. Not a second
     * source of truth: aec_config_resolve_delay() folds it into delay_mode
     * once, before validation, and the rest of the library reads delay_mode
     * alone. Mapping table and conflict rule: aec_config_resolve_delay(). To
     * be removed once all callers use delay_mode. */
    int    enable_delay_est;       /* 1 (deprecated; see delay_mode) */
    int    enable_highpass;        /* 1 (mic only) */
    int    enable_saturation;      /* 1 */
    int    enable_shadow;          /* 1 */
    int    enable_res;             /* 1 */
    int    saturation_softclip_ref;/* 1 */
    /* Linear-AEC + external-RES seam (Python AecConfig.return_res_context).
     * When set, the post-filter block RUNS (computes res_gain / R^2 /
     * comfort_noise / windowed error spectrum and exposes them via
     * AecResContext) but, if enable_res==0, its suppression is NOT applied and
     * the time output stays the linear residual (raw_output). Lets a caller
     * run AEC(linear) -> NR -> RES externally. Default 0. */
    int    return_res_context;     /* 0 */
    /* Level 1 (AEC_CONTEXT_NO_GAIN): this lane's own SuppressionGain output
     * (G_res) is never computed, because a downstream fused/beamformed
     * post-stage (e.g. a multi-channel wrapper) recomputes an equivalent gain
     * from combined multi-lane data. The DominantNearend hold-state (which the
     * next hop's ERLE onset decision, hence r2, depends on) is still updated
     * every hop; comfort_noise is computed independently (one of DNE's inputs,
     * not affected by it). Valid only when enable_res==0 &&
     * return_res_context==1 (the context-only seam); aec_validate_config()
     * rejects any other combination, since apply_output(), the only consumer
     * of the skipped gain, runs only when that seam is NOT active. Default 0.
     *
     * Level 2 (AEC_CONTEXT_LINEAR_ONLY) goes further, for a caller that reads
     * only the linear half of AecResContext / AecLinearContext (formed_hop,
     * error_spec, echo_spec, near_spec, far_spec, the delay and verdict
     * fields): the post block stops after the AecState update. The
     * reverb/residual estimator, the stationarity R^2 zeroing, comfort-noise
     * tracking and the suppression gain exist only to feed res_gain / r2 /
     * comfort_noise, and nothing downstream of them feeds the linear filter,
     * so every linear output is bit-identical to levels 0 and 1. res_gain, r2
     * and comfort_noise read NULL and res_floor_protect reads 0. Same
     * validity rule as level 1. */
    int    spatial_linear_context; /* 0 */

    /* Scalar tunables (balanced base). shadow_err_alpha / warmup_frames /
     * epc_hangover are authored on the hop=160 / sr=16000 (10 ms) grid and
     * retimed in aec_carve(), the one place guaranteed to see the final
     * resolved hop after any caller override (e.g. aec_wav.c --fft-size), not
     * in aec_config_defaults(). The values below are reference-grid figures
     * and may differ from the effective values an Aec stores. */
    float  shadow_err_alpha;       /* 0.80 @ 10 ms hop */
    float  shadow_mu_min;          /* 0.5  */
    float  shadow_mu_nlms;         /* 0.5  */
    int    warmup_frames;          /* 100 hops = 1000 ms @ 10 ms hop */
    int    epc_hangover;           /* 20 hops = 200 ms @ 10 ms hop */
    float  epc_total_rise;         /* 1.5  */
    float  epc_delta_threshold;    /* 0.3  */
    float  epc_mu_floor;           /* 0.5  */
    /* MATCHED only: together these size the reference SEARCH ring,
     * max(delay_buffer_ms, max_delay_ms + 4096 samples). Inert under FIXED
     * (sized from fixed_delay_samples + hop) and EXTERNAL_ALIGNED (no ring);
     * see aec_ref_ring_samples() in aec.c. */
    float  max_delay_ms;           /* 1024 */
    float  delay_buffer_ms;        /* 2048 */
    float  delay_est_init_s;       /* 0.3  */
    float  delay_est_period_s;     /* 0.0 (full-rate) */
    /* Matched-filter bank size in [1, DA_NUM_FILTERS] (Python
     * AecConfig.delay_num_filters). 5 is the default geometry and the only
     * value bench/dataset runs may use. Smaller is a compute and memory knob
     * for an embedded target whose bulk system delay is already compensated
     * out-of-band, leaving the matched filter only the residual to track:
     * the reliable reach drops to 125/221/317/413/509 ms for n=1..5, each
     * dropped filter removes ~4.2 MMAC/s of full-rate search, and
     * aec_get_mem_size() drops by exactly 5,728 B per filter on every grid
     * (the bank, render ring and both lag histograms are pool-carved to this
     * n; delay_aec.h's MEMORY CONTRACT block). It is part of the POOL LAYOUT:
     * init-time immutable like the signal grid, so changing it means
     * re-querying aec_get_mem_size() and re-initialising.
     * Out-of-range values are REJECTED by aec_validate_config, and so is any
     * non-default value when delay_mode is not MATCHED (no matched filter
     * exists to size, so it would be a silently ignored input). */
    int    delay_num_filters;      /* 5    */
    float  highpass_cutoff_hz;     /* 80   */
    float  saturation_threshold;   /* 0.95 */
    float  kalman_q_high;          /* 1e-3 */
    float  kalman_q_low;           /* 1e-6 */
    int    delay_acquire_protect_converged;   /* 1 */
    /* Warm tap-transfer on first delay acquisition (default ON): shift the
     * learned IR by the acquired delay instead of zeroing, so cold-start
     * cancellation survives the realign. Gated on inst-ERLE peak >
     * delay_acquire_inst_erle_db AND delay < tap reach. */
    int    delay_acquire_warm_transfer;       /* 1 */
    float  delay_acquire_inst_erle_db;        /* 4.0 */

    /* DT-deg recovery stack (default ON, as in Python). The energy-based held
     * near-end gate (_ne_recent_frames) arms two levers during double-talk
     * only (FS never raises dt_from_energy): */
    int    dt_aware_recovery_soft;            /* 1 — soft (non-destructive)
                                               * Path A/B/EPV/shadow_rise realign
                                               * while near recently present */
    int    dt_aware_res_floor_enabled;        /* 1 — DT-gated RES min-gain floor */
    float  min_gain_floor_dt_db;              /* -20.0 — DT floor (dB) */
    /* ne_recent gate parameters (AecConfig.ne_recent_*). threshold is float32
     * (f32/f64 drift across it versus Python is accepted). ne_recent_hold is
     * authored at the 10 ms hop and retimed in aec_carve(). ne_recent_sustain
     * is deliberately left unretimed; see config.py's field comment. */
    float  ne_recent_threshold;               /* 0.3 */
    int    ne_recent_hold;                    /* 150 hops = 1500 ms @ 10 ms hop */
    int    ne_recent_sustain;                 /* 3 (left unretimed; see comment above) */

    /* Preset strength axis, the only field mild/balanced/aggressive differ
     * in: SuppressionGain far-active min-gain floor in dB (-20 / -28 / -38). */
    float  min_gain_floor_far_active_db;

    /* FilterMisadjustmentEstimator (filter scale tracker). stable/hangover
     * frames are authored at the 10 ms hop and retimed in aec_carve(). */
    int    filter_misadjustment_stable_frames;   /* 30 hops = 300 ms @ 10 ms hop */
    int    filter_misadjustment_hangover_frames;  /* 100 hops = 1000 ms @ 10 ms hop */
    float  filter_misadjustment_scale_min;        /* 0.5 */
    float  filter_misadjustment_scale_max;        /* 2.0 */

    /* ── delay mode (appended at the end so every field above keeps its byte
     * offset) ──────────────────────────────────────────────────────────────
     * The single source of truth for far/mic alignment (AecDelayMode).
     * Init-time immutable: changing it needs destroy/re-init, since the pool
     * layout differs per mode. Default AEC_DELAY_MATCHED. */
    AecDelayMode delay_mode;       /* AEC_DELAY_MATCHED */
    /* FIXED only: the measured system delay in NATIVE-rate samples, >= 0. It
     * must be < 0 (the -1 "unset" sentinel) in MATCHED and EXTERNAL_ALIGNED; a
     * fixed delay those modes cannot honour is rejected, not ignored (Python
     * AecConfig.fixed_delay_samples). It sizes the reference ring exactly:
     * FIXED carves `fixed_delay_samples + hop_size` samples and ignores
     * max_delay_ms / delay_buffer_ms (those size the MATCHED SEARCH ring) --
     * the rule of config.py ref_ring_samples(). */
    int    fixed_delay_samples;    /* -1 */

    /* ── Live observer: a second, independent PBFDAF filter reading the
     * far-end reference ring earlier-shifted than the main filter's own
     * delay-compensated read. It feeds extra residual-echo evidence into R^2
     * only while its residual corroborates against the main filter's (never
     * against its own past state; rationale in aec_post.h's AecPostConfig
     * observer_* fields). Mirrors config.py live_observer_*. Default OFF,
     * byte-equal when off. Appended at the end, like delay_mode, so earlier
     * fields keep their byte offsets. */
    int    live_observer_enabled;           /* 0 */
    float  live_observer_shift_ms;          /* 24.0 */
    float  live_observer_max_hz;            /* 1000.0 */
    float  live_observer_mu;                /* 0.3 */
    float  live_observer_corrob_ms;         /* 100.0 */
    float  live_observer_corrob_margin_db;  /* 0.0 */

    /* ── NO_FLOOR preset support, appended at the end for the same
     * ABI-stability reason (earlier fields keep their byte offsets). Mirrors
     * Python AecConfig.min_gain_split_floor_enabled: gates ALL THREE floor
     * variants in suppression_gain.c (far-active, far-silent/pure-NE,
     * DT-lifted), not just the far-active dB value that mild / balanced /
     * aggressive vary. Default 1, which is also what aec_config_defaults()
     * fills for a caller that never goes through a preset. */
    int    min_gain_split_floor_enabled;    /* 1 */
} AecConfig;

void aec_config_defaults(AecConfig* cfg, int sample_rate);
void aec_config_from_preset(AecConfig* cfg, AecPreset preset, int sample_rate);

/* ── Deprecated-field translation layer (short-term compatibility) ────────
 * Folds the deprecated `enable_delay_est` mirror into `delay_mode`, in place,
 * so afterwards `delay_mode` alone decides alignment behaviour. Idempotent;
 * called automatically by aec_validate_config / aec_get_mem_size /
 * aec_create / aec_init, so a caller never has to invoke it. It is public so
 * tests and integrators can inspect the resolved mode without constructing
 * an Aec.
 *
 * Mapping (the only three legacy shapes that ever existed):
 *
 *   enable_delay_est | fixed_delay_samples | delay_mode in | resolved mode
 *   -----------------+---------------------+---------------+---------------
 *    1 (default)     | any                 | any           | delay_mode (unchanged)
 *    0               | >= 0                | MATCHED (dflt)| FIXED
 *    0               | <  0                | MATCHED (dflt)| EXTERNAL_ALIGNED
 *    0               | any                 | FIXED         | FIXED
 *    0               | any                 | EXTERNAL_ALIGN| EXTERNAL_ALIGNED
 *
 * `enable_delay_est == 1` (its default) carries no information and leaves
 * delay_mode alone; clearing it is the legacy way of saying "not MATCHED" and
 * translates only while delay_mode is still at its default. A caller that
 * sets BOTH already agrees with itself, since the only non-default delay_mode
 * values are the two non-MATCHED outcomes the legacy field can select.
 *
 * On return `enable_delay_est` is rewritten to the mirror of the resolved
 * mode (1 for MATCHED, 0 otherwise), so a round-tripped config is
 * self-consistent whichever field a caller reads.
 *
 * Returns 0 on success, -1 if cfg is NULL or delay_mode is not one of the
 * three enumerators. */
int aec_config_resolve_delay(AecConfig* cfg);

/* ── Signal grid: one resolver, shared by every entry point ───────────────
 * (sample_rate, fft_size) completely describes the frame geometry; the rest
 * is derived. This project fixes frame_size == fft_size and
 * hop_size == fft_size/2, and that derivation lives only in
 * aec_resolve_signal_grid(), through which aec_validate_config(),
 * aec_get_mem_size(), aec_create() and aec_init() all go, so a caller can
 * never be sized against one grid and initialised against another.
 *
 * fft_size == 0 ("auto") is REJECTED: 16 kHz has TWO production grids (256
 * and 512) and a library that guesses would silently pick one. The
 * convenience factories aec_config_defaults() / aec_config_from_preset() fill
 * a concrete fft_size (16k -> 256, 48k -> 1024, 8k -> 256) and are the only
 * place a default is applied. A CLI may pick a product default for a user who
 * gave none, but must resolve it to a nonzero fft_size before calling in.
 *
 * Supported grids:
 *
 *   sample rate | fft/frame | hop | status
 *   ------------+-----------+-----+------------------------------------
 *    16000      |  256      | 128 | product (default at 16 kHz)
 *    16000      |  512      | 256 | product (alternate, higher compute)
 *    48000      | 1024      | 512 | product
 *     8000      |  256      | 128 | LEGACY -- supported, NOT a product
 *                                 | grid; kept for the 8 kHz E2E-parity /
 *                                 | structural tests and the Audio_ALG MONO
 *                                 | pipeline. The 4-channel pipeline
 *                                 | contracts to the three product grids
 *                                 | only. is_legacy is set for it. */
typedef struct AecSignalGrid {
    int sample_rate;
    int fft_size;     /* == frame_size in this project */
    int frame_size;
    int hop_size;     /* == fft_size / 2 */
    int n_freqs;      /* == fft_size / 2 + 1 */
    int is_legacy;    /* 1 for 8 kHz: supported, not a product grid */
} AecSignalGrid;

/* Resolve (sample_rate, fft_size) into the full grid. Returns 1 and fills
 * *out on a supported pair; returns 0 and leaves *out untouched otherwise
 * (including fft_size == 0). `out` may be NULL for a pure validity query. */
int aec_resolve_signal_grid(int sample_rate, int fft_size, AecSignalGrid* out);

/* Sample-rate whitelist: is there at least one supported grid at this rate.
 * Reads the same table as aec_resolve_signal_grid(), for callers (e.g.
 * example/aec_wav.c) that must reject an unsupported WAV rate before choosing
 * an fft_size. Returns 1 for 8000/16000/48000. A true result does NOT mean
 * any fft_size will do: the pair must still resolve. */
int aec_is_valid_sample_rate(int sample_rate);

/* The product-default fft_size for a rate, or 0 if the rate is unsupported.
 * The implementation behind aec_config_defaults()'s fill; exposed so a CLI
 * can resolve a user's unspecified grid to a concrete value instead of
 * hard-coding its own copy of the mapping. */
int aec_default_fft_size(int sample_rate);

/* Streaming render/capture buffering events, returned by the streaming API. */
typedef enum AecBufferingEvent {
    AEC_BUF_NONE = 0,
    AEC_BUF_RENDER_UNDERRUN,   /* capture ran ahead — no buffered render */
    AEC_BUF_RENDER_OVERRUN,    /* render piled up past the FIFO — oldest dropped */
    AEC_BUF_API_SKEW           /* render/capture call alternation anomaly (diag) */
} AecBufferingEvent;

typedef struct Aec {
    AecConfig cfg;

    int hop_size, block_size, fft_size, n_freqs, n_partitions;

    /* preprocessing */
    Hpf  *hp_mic;             /* NULL when highpass disabled */
    Saturation sat_ref, sat_mic;  int has_sat;

    /* `delay` is a metadata struct: its arrays (bank, render ring, both lag
     * histograms, 48 kHz sidechain scratch) live in this instance's pool,
     * carved by aec_carve() at the size aec_get_mem_size() budgeted for the
     * resolved (sample_rate, hop, delay_num_filters) triple, never at a
     * compile-time maximum. Only MATCHED carves it; when has_delay == 0 every
     * pointer inside is NULL and every size 0. has_delay == 1 iff
     * cfg.delay_mode == AEC_DELAY_MATCHED, i.e. iff the matched-filter
     * ESTIMATOR was constructed (Python `delay_est is not None`). The
     * reference RING is a separate question: it exists for MATCHED and FIXED,
     * and only EXTERNAL_ALIGNED has none (Python `_delay_active`). Ask
     * aec_ring_active(&cfg) (aec.c) whether there is a ring, never has_delay
     * and never a re-spelled delay_mode comparison, so the two questions
     * cannot merge. */
    DelayAec delay;       int has_delay;
    float* ref_ring;       int ref_ring_size, ref_ring_write, ref_ring_filled;
    int    current_delay;  /* MATCHED: -1 until first acquisition.
                            * FIXED:   cfg.fixed_delay_samples, always.
                            * EXTERNAL_ALIGNED: 0 (nothing to apply). */
    int    pending_delay;  int has_pending; int pending_delay_ttl;
    /* C mirror of the Python bad-relock judgment. The matched-filter
     * aggregator (shared design in both languages) accepts whichever lag
     * correlates best, with no check that it actually cancels better. Path B
     * therefore judges a shift AEC_DELAY_SHIFT_JUDGMENT_HOPS after it commits
     * and reverts if erle_windowed is more than 1 dB worse than just before
     * (see aec_process_core's Path B block for why the window is 100, not
     * delay_est_init_s).
     * judgment_active: 0 = no pending judgment; 1 = counting down.
     * judgment_pre_delay / _pre_erle: the ring alignment and erle_windowed from
     * just before the judged shift, to revert to / compare against.
     * judgment_hops_left: counts down to 0, where the verdict fires (the
     * hard-reset chain zeroes W first and the snapshot restores it after,
     * never before).
     * judgment_pre_W / _pre_shadow_W: pool-carved snapshots of
     * main_filter.W / shadow_filter.W at the moment the shift committed, the
     * filters' own shape (np*K Complex each), carved right after the live
     * observer in aec_get_mem_size() / aec_carve(); the shadow's only when
     * cfg.enable_shadow. */
    int    delay_shift_judgment_active;
    int    delay_shift_judgment_pre_delay;
    float  delay_shift_judgment_pre_erle;
    int    delay_shift_judgment_hops_left;
    Complex* delay_shift_judgment_pre_W;
    Complex* delay_shift_judgment_pre_shadow_W;
    /* Confirmed-bad relock candidates, remembered for the instance's life
     * (cleared only by aec_reset()) so a bad lag gets one correction attempt,
     * not an infinite retry loop. Mirrors Python's _delay_reject_values except
     * for the cap: Python's list is unbounded, this is a fixed
     * AEC_DELAY_REJECT_CAP-slot array because zero-heap static memory has no
     * unbounded container. It saturates (stops recording) rather than
     * overwriting the oldest entry, since an overwrite could un-reject a value
     * that is still bad, and reaching the cap means the session is far outside
     * anything observed in testing. */
    int    delay_reject_values[AEC_DELAY_REJECT_CAP];
    int    delay_reject_count;
    /* Alignment-generation token for external consumers (AecLinearContext).
     * Bumped (saturating, never wrapping) at EVERY write that changes the
     * alignment the filter is matched to: first acquisition, confirmed shift
     * (soft and hard recovery share the write site), aec_reset(), and both
     * outcomes of aec_apply_external_realign() (the only bump
     * EXTERNAL_ALIGNED has, since it owns no ring offset to move). Pollers
     * cannot reconstruct it from delay_samples differences: the soft-recovery
     * paths deliberately set no other flag, and in EXTERNAL_ALIGNED
     * delay_samples is 0 by contract. */
    unsigned int delay_generation;
    unsigned int delay_gen_hop_start;  /* snapshot at hop entry (CHANGED detect) */
    int    far_hop_aligned;    /* 1 = this hop's far_hop was ring-compensated
                                * (or the applied delay is exactly 0); 0 = raw,
                                * unaligned far (pre-acquisition / ring not yet
                                * filled / delay estimation disabled) */
    /* Delay-estimator duty-cycle state. K = max(1, round(delay_est_period_s
     * /hop_s)/5). With delay_est_period_s at its default 0.0, K=1 is
     * full-rate, matching the Python reference (no throttle). A value > 0 (or
     * --delay-est-period) opts into self-gating the matched-filter analysis to
     * 1 hop in every K once the estimate is solid (confidence 1.0) AND
     * unchanged for delay_est_init_s seconds, e.g. K=10 at a 0.5 s period @
     * 10 ms hop, cutting ~90% of the matched-filter cost at the price of the
     * estimator finding different candidate lags than Python on the same
     * audio (confirmed on a real file, not just a timing difference). The
     * estimator's decimators and render ring stay FED every hop regardless of
     * K (only the analysis decimates), so the matched filter never sees gapped
     * audio. When K>1, full-rate analysis resumes immediately when (a) the
     * estimate changes or loses solidity, or (b) the ERLE watchdog fires:
     * erle_windowed drops >6 dB below its running peak (the peak leaks 0.1 dB/s
     * at every grid; armed only once it exceeds 6 dB, so it cannot fire before
     * first convergence).
     *
     * The aggregator weakness behind a bad relock is caught by the
     * judgment+revert mechanism above, not by this watchdog. With
     * delay_est_period_s at 0.0, duty_active never arms in production, so the
     * watchdog has nothing to watch by default. */
    int    duty_active;        /* 1 = decimated analysis in effect */
    int    duty_stable_hops;   /* consecutive solid+unchanged hops (arming) */
    int    duty_pos;           /* position in the 1-in-K analysis cycle */
    int    duty_last_delay;    /* estimate on the previous hop (change detect) */
    float  duty_erle_peak;     /* leaky running peak of erle_windowed (dB) */
    float  duty_erle_leak_db;  /* per-hop leak of duty_erle_peak (dB), derived
                                * from the grid at carve time so the wall-clock
                                * rate stays 0.1 dB/s. Sits in what was tail
                                * padding, so sizeof(Aec) and later offsets are
                                * unchanged.                                  */
    /* Engagement census. The ~90% matched-filter saving above is a DESIGN
     * figure that only materialises on hops the duty machine decimates; on an
     * echo path whose estimate keeps moving it re-arms every time and never
     * decimates. These two counters make the REALISED saving measurable:
     * duty_hops_run / duty_hops_total is the true matched-filter engagement,
     * and 1 minus that ratio the true saving. Incremented at the single call
     * site that consumes run_filter (aec_process_core), so the census cannot
     * drift from what ran. uint64 because a 128-sample hop passes 2^31 hops in
     * ~2 years of continuous audio. Diagnostic only (nothing in the library
     * reads them back); exposed read-only via AecDebugStatus and printed by
     * example/aec_wav.c. Cumulative since aec_create/aec_init/aec_reset. */
    unsigned long long duty_hops_total;  /* hops through the delay block   */
    unsigned long long duty_hops_run;    /* of those, hops that actually
                                          * ran the matched filter
                                          * (run_filter == 1)               */

    /* linear filters */
    PBFDKF main_filter;
    PBFDAF shadow_filter;  int has_shadow;
    /* Live observer: a third PBFDAF built like the shadow filter (static-init
     * against the shared post_fft, same seeding/reset mirrors; see aec_carve()'s
     * `cfg->live_observer_enabled` block). Unlike the shadow it is driven from
     * INSIDE aec_post_run, at the hop where Python's post-filter step runs its
     * live-observer block, because it reads this hop's dominant_ne /
     * usable_linear verdicts, known only there. */
    PBFDAF observer_filter;  int has_observer;

    /* detectors / EPC / regime / RSA */
    RenderActivity    render_activity;
    /* RenderActivity scratch: pool-carved, ceil(hop_size/8) (detectors.h). */
    float*            ra_pairwise_scratch;
    FilterConvergence convergence;
    DoubleTalk        dt_analyzer;
    EpcDetector       epc;
    ShadowCopy        regime;
    RenderSignalAnalyzer rsa;   int64_t* rsa_counters;

    /* post-filter chain (driver + sub-module objects, all caller-owned storage) */
    FftHandle*  post_fft;
    AecPost    post;
    AecState              a3_state;   AecStateStorage a3_state_st;
    ResidualEchoEstimator a3_ree;
    SuppressionGain       a3_sg;
    StationarityEstimator a3_stat;
    LinearFilterSelect    a3_lfs;
    AecPostRunScratch    a3_sc;
    /* Pending EPV mirror + stationarity read-state */
    int    pending_gain_change;   /* bool */
    int    pending_delay_change;  /* -1 == None, else DelayAdjustment */
    int    stationarity_active_hops;
    int    stationarity_converge_hops;
    int    non_zero_render_seen;
    float  render_peak_floor;
    int    block_stationary_next;   /* _block_stationary_for_next_hop latch */

    /* per-frame scalars (mirror the Python orchestrator instance fields); float32. */
    float  saturation_level;
    float  erl_estimate;
    float  main_err_smooth, shadow_err_smooth;
    long   shadow_frame_count;     /* saturates at LONG_MAX */
    int    epc_render_forced_remaining;
    float  erle_window_near, erle_window_err;
    float  erle_factor_prev;
    float  inst_erle_smooth;
    /* inst-ERLE slope ring (Python _erle_slope_buf, a deque of ~500 ms/hop).
     * Holds get_erle_instant() dB values; the warm tap-transfer gate reads the
     * max of the last 15. Filled every freq-path hop OUTSIDE the enable_res
     * gate (Python appends unconditionally). */
    float  erle_slope_buf[64];   /* cap >= ceil(500ms/hop)=50 @hop160/16k */
    int    erle_slope_cap, erle_slope_len, erle_slope_head;
    float  wn_err_baseline;
    int    stat_dt_hangover;
    int    warmup_frames_remaining;
    int    warmup_far_active;
    float  simple_mu_ratio;
    int    simple_mu_holdoff;
    float* per_bin_mu_scale;   int has_per_bin_mu;
    /* power EMAs (alpha=0.95 sample loop) */
    float  near_power, raw_error_power;
    float  alpha_pow;
    long   frame_count;            /* saturates at LONG_MAX */

    /* DT-deg recovery: held "near-end seen recently" gate (Python _ne_above
     * consecutive-above counter and _ne_recent_frames hold countdown). */
    int    ne_above;
    int    ne_recent_frames;

    /* poor-coarse rescue (process loop) */
    int    poor_coarse_counter;
    int    coarse_reset_hangover;
    /* Track F: sustained leakage-diverged gate counter (clamped 0 ..
     * 2*leakage_div_sustain_hops), Python _leakage_div_sustained_counter. */
    int    leakage_div_sustained_counter;
    /* Thresholds for the counters above (and stat_dt_hangover), recomputed for
     * the live hop_size/sample_rate; Python uses blocks_to_hops(5)/(25) and
     * ms_to_hops(50.0)/(800.0) at the same sites. */
    int    poor_coarse_threshold_hops;
    int    coarse_reset_hangover_hops;
    int    leakage_div_sustain_hops;
    int    stat_dt_hangover_hops;

    /* Matched-delay duty-cycle gate (aec_process_core's Path-A analysis skip):
     * lrintf(delay_est_init_s/hop_s) and lrintf(delay_est_period_s/hop_s)/5,
     * post-clamp, cached because hop_size, sample_rate and the two cfg seconds
     * fields are fixed for the instance's lifetime. Named distinctly so
     * duty_period_hops does not shadow the function-local `K` (= n_freqs). */
    int    duty_hold_hops;
    int    duty_period_hops;

    /* FilterMisadjustmentEstimator accumulators */
    float  misadj_e2_acum, misadj_y2_acum;
    int    misadj_n_acum;
    float  misadj_inv;
    int    misadj_overhang;
    int    misadj_stable_count;
    int    misadj_hangover_remaining;

    /* erle_windowed cache for the delay Path-A _already_cancelling guard
     * (last frame's erle_windowed dB). */
    float  last_erle_windowed;

    /* RES-context stash (last frame) */
    float  last_far_power, last_shadow_dt, last_dt_indicator;
    /* Retimed ERL tracker EMAs (per hop, authored at the 10 ms grid). */
    float  alpha_erl_tracking, alpha_erl_converged;
    int    last_is_stationary_dt;

    /* hop scratch */
    float* near_hop;
    float* far_hop;
    float* raw_output;
    float* shadow_out;
    float* final_out;
    /* Live observer only (has_observer): the far-end ring read shifted
     * earlier than far_hop's own delay-compensated read (see the ring-read
     * block beside far_hop's in aec.c). Written here because the ring is owned
     * by this struct; read via AecPostRunIn.observer_far_end. The observer's
     * own per-hop scratch (discarded time-domain output, corroboration
     * magnitude arrays) lives in AecPostRunScratch with sel_esw/r2/etc.,
     * since only aec_post_run touches it. */
    float* observer_far_hop;
    int    observer_far_hop_valid;
    int    observer_shift_hops;   /* ms_to_hops(live_observer_shift_ms), carve-time */
    float* filter_taps_full;   /* [n_partitions × hop] */

    /* Per-hop freq-bin scratch, kept in the Aec instance (malloc'd in
     * aec_create, pool-carved in aec_init) and sized exactly K (hop for
     * scr_sq, which mean_sq() runs only over hop-length buffers) instead of
     * fixed-size locals, so stack use does not depend on n_freqs. A fixed
     * 8192-entry local (some scopes stacking 2-3, ~96 KB worst case) would be
     * unsafe for an embedded RTOS task stack (often 16-64 KB in total). */
    float* scr_sq;          /* mean_sq() scratch, size hop_size (not K) */
    float* scr_e2_echo;
    float* scr_e2_near;
    float* scr_rsa_mask;
    float* scr_mu_buf_pre;
    float* scr_e2coa_pre;
    float* scr_mu_buf;
    float* scr_e2ref_arr;
    float* scr_erl_arr;

    /* ── streaming render-hop FIFO (async render/capture decoupling) ──
     * Used only by the streaming API (aec_analyze_render /
     * aec_process_capture). Lockstep aec_process() bypasses it, leaving offline
     * byte-exact parity with Python untouched; in lockstep use (one
     * analyze_render then one process_capture) it is pass-through (fifo_write
     * one ahead of fifo_read, no event) and the output equals aec_process().
     *
     * SPSC ring with one writer per cursor. On overrun the producer drops the
     * new hop; when a full ring separates the cursors, the consumer catches up
     * to the freshest hop. Cross-thread cursor access uses GCC/Clang __atomic
     * builtins on plain fields so the public layout stays C89-compatible.
     * Per-field protocol:
     *   - fifo_write: producer-owned monotonic unsigned write sequence number,
     *     indexed into the ring with `% cap` at each use (never stored
     *     pre-wrapped), so unsigned wraparound is well-defined and harmless.
     *     Sole writer: the render thread, in aec_analyze_render(), which reads
     *     it as a plain load (no other writer exists, so it cannot race) and
     *     RELEASE-stores it only after the new hop's memcpy into the ring slot
     *     has completed, publishing the payload to whichever thread's ACQUIRE
     *     load of fifo_write next sees the incremented value.
     *   - fifo_read: consumer-owned monotonic unsigned read sequence number,
     *     same wraparound treatment. Sole writer: the capture thread, in
     *     aec_process_capture(), including its catch-up path (the producer
     *     never touches it). Plain-loaded by its own thread; RELEASE-stored
     *     only AFTER aec_process() has returned for the claimed hop, which both
     *     publishes the cursor and marks the slot free to the producer's next
     *     ACQUIRE load of fifo_read.
     *   - occupancy is always `fifo_write - fifo_read` (unsigned subtraction).
     *     fifo_cap_hops is forced to a power of two (aec_derive_dims in aec.c)
     *     so this stays correct across the 2^32 wrap of either cursor:
     *     `(x mod 2^32) mod cap == x mod cap` for every unsigned x only when
     *     cap divides 2^32. The invariant `0 <= fifo_write - fifo_read <= cap`
     *     holds at every instant by induction: the producer advances fifo_write
     *     only after observing `fifo_write - fifo_read_obs < cap` for some
     *     fifo_read_obs <= the true fifo_read (an ACQUIRE load can only lag its
     *     writer), so the post-increment true occupancy is <= cap; the consumer
     *     advances fifo_read only after observing `fifo_write_obs - fifo_read >
     *     0` for some fifo_write_obs <= the true fifo_write, so the
     *     post-increment true occupancy is >= 0. Slot aliasing (two live claims
     *     on one physical index) would need fifo_write - fifo_read equal to 0
     *     or cap, exactly the boundaries the guards forbid crossing without the
     *     matching drop-new / catch-up event, so the producer never writes a
     *     slot the consumer may still read and the consumer never reads a slot
     *     the producer has not fully written.
     *   - The producer's occupancy check is staleness-safe in the conservative
     *     direction: fifo_read_obs <= the true fifo_read, so the computed
     *     occupancy can only OVER-estimate (worst case a spurious drop-new when
     *     there was room), never under-estimate, so it never overwrites a slot
     *     the consumer has not released.
     *   - Slot payload: the producer memcpy's the new hop into the ring BEFORE
     *     its RELEASE store of fifo_write; the consumer RELEASE-stores its
     *     advanced fifo_read only AFTER aec_process() has returned (whose first
     *     act is to memcpy the passed-in `ref` OUT of the ring into
     *     a->far_hop), so the slot stays "claimed" for the whole hop. That is
     *     more conservative than necessary (it can turn one extra producer call
     *     into a reported drop-new under sustained skew) but needs no change
     *     inside aec_process().
     *   - fifo_count: always 0 and unused; kept so later fields keep their
     *     byte offsets. aec.c only zeroes it at init/reset.
     *   - fifo_zero_ref: an immutable all-zero hop, carved from the pool at init
     *     and re-zeroed at reset, otherwise never written by either streaming
     *     thread. It is the underrun reference; the capture thread never
     *     writes the ring, so aec_process_capture() has nothing to race
     *     against on that path.
     *   - render_call_count / capture_call_count / last_buffering_event: each is
     *     written by exactly one side (render_call_count by the render thread,
     *     the other two by the capture thread), so plain increments are
     *     race-free by construction. last_buffering_event additionally uses a
     *     relaxed atomic store (write) and load (aec_last_buffering_event()
     *     read) so a third polling thread is not a formal data race; the SPSC
     *     contract promises no fresher than "as of the last capture call this
     *     thread happens to observe".
     * aec_reset()/aec_create()/aec_init() write these fields as plain ints,
     * correct only because the caller must fully serialize with both the
     * render and capture threads before resetting or destroying a
     * streaming-mode Aec (contract text below). The design is TSan-clean with
     * zero suppressions (test/test_fifo_spsc.c header comment). */
    float* render_fifo;        /* [fifo_cap_hops × hop_size] ring of render hops */
    int    fifo_cap_hops;      /* capacity in hops; forced to a power of two (aec_derive_dims) */
    int    fifo_count;         /* always 0; layout stability */
    int    fifo_read, fifo_write;  /* monotonic unsigned cursors: fifo_read consumer-owned, fifo_write producer-owned */
    long   render_call_count, capture_call_count;  /* single-writer-per-field stats, plain ops; each saturates at LONG_MAX */
    int    last_buffering_event;   /* AecBufferingEvent of the last capture step; relaxed-atomic */

    int is_static;

    /* Single allocation owned by aec_create(); NULL for caller-pool aec_init(). */
    void*  heap_arena;

    /* Immutable all-zero reference used on streaming FIFO underrun. */
    float* fifo_zero_ref;

    /* Per-hop stage timing (AecStageTiming below), appended at the end so
     * earlier fields keep their offsets. */
    struct AecStageTiming {
        uint32_t delay_us;
        uint32_t frontend_us;
        uint32_t linear_us;
        uint32_t res_us;
        uint32_t steering_us;   /* appended after the four above, keeping their offsets */
    } last_timing;
} Aec;

/**
 * Per-hop wall-clock cost of the five largest stages inside aec_process*(), in
 * microseconds. Diagnostic only: nothing in the engine reads these back and
 * they do not affect processing.
 *
 *   delay_us     delay estimation and ring alignment: the matched-filter bank,
 *                the duty-cycle bookkeeping around it, the reference ring write
 *                and the delay-compensating read. Zero under
 *                AEC_DELAY_EXTERNAL_ALIGNED (no ring, no estimator) and small
 *                under AEC_DELAY_FIXED (ring, but no matched filter). Under
 *                AEC_DELAY_MATCHED it is frequently the largest single stage,
 *                which is why it is reported apart from frontend_us.
 *   frontend_us  the rest of the path from entry up to the main filter: mic
 *                HPF, saturation, render activity, mu_scale, mic-clip, RSA
 *                update and the shadow filter. EXCLUDES delay_us, so the two
 *                never double-count.
 *   linear_us    the main adaptive filter (pbfdkf_process), including the
 *                far-end FFT when this instance computes its own.
 *   steering_us  everything between the main filter and the post block: the
 *                stationarity refresh, the e2_coarse/ERL publish and coarse
 *                rescue, the double-talk analyzer, EPV, shadow_rise and the
 *                misadjustment estimator. It never touches the sample stream
 *                directly (it decides how the next hop adapts) but is real
 *                per-hop cost, reported so the breakdown covers the whole hop.
 *   res_us       the post-filter/RES block. It runs whenever enable_res OR
 *                return_res_context is set, so a context-only caller pays it
 *                too and sees a real number.
 *
 * The five do NOT sum to the call: the remainder inside aec_process*() is the
 * per-sample power EMAs and convergence detection, which run after res_us
 * closes. A caller presenting a breakdown must carry that remainder
 * explicitly.
 *
 * Cleared at the top of every aec_process*() call, so a hop reports only what
 * it reached. Resolution is microseconds: a stage under 1 us reads 0.
 *
 * OFF unless built with -DAEC_STAGE_TIMING=1. Measuring costs seven clock reads
 * per hop PER INSTANCE (a multi-instance caller multiplies that and should
 * document its own topology). A release build reads no clock, so every field
 * is 0 on every hop and only a few scalar stores into an unread struct remain.
 * That zero is how a caller tells a build that does not measure from one that
 * does; a consumer wanting a breakdown must build the LIBRARY with the flag,
 * not just enable its own display.
 *
 * The stamp is CLOCK_MONOTONIC, which is POSIX rather than C99. A target whose
 * libc lacks it combines the flag with -DAEC_NOW_US=<fn> to name its own
 * microsecond timer; the stamp's comment in aec.c states the rules that
 * substitute must meet.
 */
/* Defined here, not in aec.c, so a CONSUMER can write `#if AEC_STAGE_TIMING`
 * against a defined macro; otherwise the test works only by an undefined
 * identifier evaluating to 0, which is silently wrong under -Wundef. */
#ifndef AEC_STAGE_TIMING
#define AEC_STAGE_TIMING 0
#endif

typedef struct AecStageTiming AecStageTiming;

/* Copies the last hop's stage timings into `out`. A NULL instance zeroes
 * `out` rather than failing; `out` must not be NULL. */
void aec_get_last_timing(const Aec* a, AecStageTiming* out);

int  aec_create(Aec* a, const AecConfig* cfg);
void aec_destroy(Aec* a);
void aec_reset(Aec* a);

/**
 * Realign an AEC_DELAY_EXTERNAL_ALIGNED instance across a caller-side change of
 * the far alignment, WITHOUT the full reset that exposes echo again and
 * restarts the WOLA sequence (an audible discontinuity).
 *
 * delta_samples = new_alignment - old_alignment: positive when the aligned far
 * stream advances (delay grew, including the first acquisition from raw far
 * where old_alignment is 0), negative when it retards.
 *
 * When the filter is demonstrably cancelling (aec_linear_is_cancelling(), which
 * ORs windowed ERLE with the recent inst-ERLE peak, NOT the inst-ERLE ring
 * alone: that ages out across far silence and would send a perfectly good
 * filter down the reset path) and the shift fits the tap span, the learned IR
 * is shifted by delta and cancellation survives the realign.
 *
 * Otherwise the filter is RESET (taps, far spectra, far analysis history and
 * the convergence/mu/stationarity latches), because taps that were not shifted
 * sit at the alignment the caller just abandoned: they do not "re-adapt from a
 * head start", they subtract a decorrelated copy of the echo (1.3-1.5x the
 * echo, worse than no filter at all, held for hundreds of hops by the
 * converged-mu floor and the stationary-render freeze). The reset is
 * FILTER-ONLY: the synthesis overlap-add and the mic-side analysis frame are
 * untouched, so the output stays continuous; that is the difference from
 * aec_reset(), whose extra reach is the synthesis restart. Reconvergence tracks
 * an aec_reset() twin's trajectory.
 *
 * Either outcome bumps the alignment generation an AecLinearContext consumer
 * polls: the far this filter is matched to moved, so cross-hop far caches are
 * stale whichever branch ran.
 *
 * Direction asymmetry. A RETARD (delta < 0) also clears the filter's
 * per-partition far spectra (the held history is ahead of the new stream, and
 * convolving it against the shifted taps mis-estimates the echo), so echo is
 * exposed until the partition ring refills past the shifted response. The
 * bound is ceil((|delta| + the response's own tap offset) / hop_size) + 1 hops,
 * at worst n_partitions + 1 when the response sits at the end of the tap span;
 * for a 256-sample hop with a 1024-sample span: delta=-300 at tap 0 -> 2 hops,
 * delta=-768 at tap 100 -> 4 hops, delta=-900 at tap 60 -> 4 hops. Within that
 * window the residual is NOT bounded by the echo either (partially refilled
 * history against the shifted taps gave 1.30x the echo on the -900 case);
 * cancellation resumes outright afterwards. An ADVANCE (delta > 0) keeps the
 * history and settles within a hop. The far samples the caller's own jump
 * skipped are unrecoverable in either direction, since they were never
 * presented to the filter.
 *
 * Returns 1 when the warm tap-transfer ran, 0 when the filter reset ran, and -1
 * on a NULL instance or a mode other than AEC_DELAY_EXTERNAL_ALIGNED.
 * delta_samples == 0 is a no-op returning 0.
 */
int aec_apply_external_realign(Aec* a, int delta_samples);

/* The strength table in one place. mild and balanced differ ONLY in the
 * far-active min-gain floor (the SuppressionGain split-floor power axis);
 * aggressive ALSO deepens the DT-lifted floor (aec_preset_dt_floor_db() below),
 * which the shared default leaves at -20 dB for mild/balanced (their far_active
 * values are already at or above -20, so a distinct dt floor would not show).
 * This is the single lookup for the far-active value.
 *
 * Public because a product can own a SuppressionGain that no Aec does (the
 * 4-channel core's shared post-stage suppressor is one) and needs the same
 * table without constructing an AecConfig to read one float.
 *
 * Returns 0 and writes *db_out for an in-enum preset, -1 otherwise. The
 * REFUSAL is the point: aec_config_from_preset() keeps its documented balanced
 * fallback for an unknown value, while every setter built on this has a caller
 * to tell. */
int aec_preset_floor_db(AecPreset preset, float* db_out);

/* Companion lookup for the DT-lifted floor (min_gain_floor_dt_db); only
 * aggressive differs from the shared -20 dB default (aec_preset_floor_db()
 * above). Same contract: 0 and *db_out written for an in-enum preset, -1
 * (nothing written) otherwise. */
int aec_preset_dt_floor_db(AecPreset preset, float* db_out);

/* Companion lookup for the gate itself (min_gain_split_floor_enabled), public
 * for the same reason as the two lookups above: the 4-channel pipeline's
 * post-beam suppressor (pipelines/4ch_aec_bf_nr_res) can read it instead of
 * only the dB value. Same contract: 0 and *enabled_out written for an in-enum
 * preset, -1 (nothing written) otherwise. */
int aec_preset_split_floor_enabled(AecPreset preset, int* enabled_out);

/* Retarget the residual-echo strength axis on a RUNNING instance.
 *
 * mild/balanced/aggressive differ in two fields (min_gain_floor_far_active_db
 * and, for aggressive only, min_gain_floor_dt_db; see
 * aec_config_from_preset); NO_FLOOR instead clears a third, the boolean
 * min_gain_split_floor_enabled. A preset change is a retarget, not a rebuild,
 * in every case: the filter, the delay state, every smoothing history, the
 * far-active latch and the DominantNearend counters all carry on.
 *
 * ramp_ms governs only the far-active floor, which the gain rule reads every
 * hop so a smooth walk is meaningful there. The DT floor is a hard per-hop
 * override (applied only while dt_protect_active), not a value the gain rule
 * converges toward, so it STEPS immediately regardless of ramp_ms.
 * min_gain_split_floor_enabled is a boolean with no continuous path to walk and
 * steps immediately too; see suppression_gain_set_split_floor_enabled() for the
 * one nuance this does not resolve (the far-active latch is not reset on the
 * gate transition). Switching into or out of NO_FLOOR needs no guard. In a
 * mid-ramp window where dt_protect_active is also true, the far-active floor is
 * still partway to its target while the DT floor has landed; the Python twin
 * matches hop-for-hop.
 *
 * ramp_ms == 0 applies the far-active floor on the next hop and lands on
 * exactly the floor a fresh instance built with `preset` would use. A positive
 * ramp_ms walks there linearly in dB, which suits an interactive strength
 * control: mild <-> aggressive is an 18 dB step and the floor is a hard clamp
 * with nothing downstream to smooth it. 100 ms is a reasonable start; it is
 * the caller's choice. ramp_ms is bounded at 60 s. A call made while a ramp is
 * running restarts from the current live value.
 *
 * 4-channel pipeline: a lane built with spatial_linear_context never calls the
 * suppressor's gain path, so this is inert on such a lane by construction; the
 * gain shaping that product's output comes from its own shared post-stage
 * suppressor, which the 4-channel core's own preset setter targets.
 *
 * Call between hops, serialised with aec_process(); not thread-safe.
 * Returns 0 on success, -1 on a NULL instance, an out-of-enum preset (this
 * entry point REFUSES where aec_config_from_preset falls back to balanced, since
 * a setter has a caller to tell) or an out-of-range ramp_ms. On -1 nothing is
 * written. */
int aec_set_preset(Aec* a, AecPreset preset, float ramp_ms);

/* Static-memory companions: place Aec and all backing arrays in one pool.
 * aec_get_mem_size() returns the total byte requirement (caller allocates
 * once); aec_init() places the Aec struct at mem[0] and returns it, with no
 * malloc. */
size_t aec_get_mem_size(const AecConfig* cfg);
Aec*   aec_init(void* mem, size_t mem_size, const AecConfig* cfg);

/* Process exactly hop_size samples: the OFFLINE / lockstep path, byte-exact to
 * the Python AEC.process(). Render and capture are supplied together. */
void aec_process(Aec* a, const float* mic, const float* ref, float* out);

/* Context-only variant of aec_process(): runs the full linear filter and
 * post-filter/RES block (so aec_get_res_context() is fully populated
 * afterwards) but never writes an `out` buffer, for callers that only read the
 * context (error_spec / res_gain / formed_output / etc.). Cheaper per hop than
 * aec_process() by exactly one O(hop) memcpy; the linear filter and RES/post
 * cost is identical.
 *
 * No mixing restriction: this, aec_process(), aec_process_capture() and
 * aec_process_context_shared_far() advance identical state and differ only by
 * whether the result is copied out, so they may be interleaved freely on one
 * instance without an intervening aec_reset(). */
void aec_process_context(Aec* a, const float* mic, const float* ref);

/* Shared-far-end-FFT variant of aec_process_context(): shared_far_spec
 * (non-NULL, length n_freqs) lets this instance skip its own far-end FFT and
 * borrow one an external caller already computed, for multi-instance callers
 * (e.g. a 4-lane wrapper) whose instances all see the identical far-end signal
 * every hop. shared_far_spec must be exactly the value the computing
 * instance's aec_get_res_context() returns this same hop, computed from the
 * SAME far-end time-domain signal this call's `ref` carries. `ref` is still
 * required: the OLA far-buffer history and every non-FFT use of the raw far
 * signal are unaffected by the sharing. Pass NULL to fall back to
 * aec_process_context() (compute this instance's own FFT). aec.c's comment on
 * this function states the full precondition. */
void aec_process_context_shared_far(
        Aec* a, const float* mic, const float* ref,
        const Complex* shared_far_spec);

/* Shares the current far spectrum AND its exact squared magnitudes from
 * far_owner, which must have completed this same hop before this call. The
 * instances must consume identical ref samples with synchronized far-buffer
 * histories; a caller with mixed realign outcomes must use the ordinary
 * context entry point on that hop. Each instance keeps its own spectrum,
 * far-magnitude history and x2 cache: previous-hop RSA reads remain local.
 * No owner pointer is retained. NULL/self/incompatible-grid owners fall back
 * to a local computation. Other stale-owner or history mismatches are the
 * caller's responsibility, as for aec_process_context_shared_far(). */
void aec_process_context_shared_far_from(
        Aec* a, const float* mic, const float* ref, const Aec* far_owner);

/* Shared-far-end-FFT instrumentation, read-only: how many hops (cumulative
 * since construction or the last aec_reset()) this instance ran its own
 * far-end FFT rather than borrowing one, whether via the internal shadow->main
 * dedup or an external aec_process_context_shared_far() caller. A lane driven
 * only through aec_process_context_shared_far() with a valid spectrum every
 * hop stays at 0; a lane computing its own (aec_process() /
 * aec_process_context()) adds 1 per hop. For tests and instrumentation proving
 * a multi-lane caller's total far-FFT count dropped after switching lanes to
 * one shared spectrum; it is outside the audio path and affects no processing.
 * The counter and sum saturate at LONG_MAX until reset. */
long aec_far_fft_real_compute_count(const Aec* a);

/* Read-only: is this instance's linear filter demonstrably cancelling right
 * now? 1 = yes.
 *
 * The single definition of "cancelling" this engine uses, exported so a wrapper
 * that owns its own shared delay estimator (pipelines/4ch_aec_bf_nr_res) asks
 * its lanes the SAME question, thresholds included, instead of restating them.
 * It reads two figures the engine already maintains (last hop's windowed ERLE
 * against 2.5 dB, and the recent inst-ERLE peak against
 * cfg.delay_acquire_inst_erle_db), so it adds nothing to the hot path.
 *
 * Both readings start at 0, so an instance whose ERLE machinery has not run
 * (or was just reset) answers 0: "not demonstrably cancelling" is the
 * fail-open answer, never a blocking one. It is evidence of cancellation, not
 * of its absence: a 0 does NOT assert the filter is broken, only that nothing
 * here proves it is working. */
int aec_linear_is_cancelling(const Aec* a);

/* ── Streaming API (real-time async render/capture) ───────────────────────
 * For deployments where the far-end (render) and mic (capture) arrive on
 * separate calls / threads and not necessarily 1:1. aec_analyze_render()
 * buffers a render hop; aec_process_capture() consumes the delay-aligned
 * render and produces output. Both return the buffering event observed.
 *
 * THREADING CONTRACT (SPSC only): exactly ONE thread may call
 * aec_analyze_render() and exactly ONE (possibly different) thread may call
 * aec_process_capture() on a given Aec (single-producer/single-consumer).
 * Under that contract the two functions may run fully concurrently with no
 * external lock: every field they share (fifo_write, fifo_read) is touched
 * only through acquire/release `__atomic_*_n` builtins (field comments on the
 * Aec struct above), and the ring's audio payload is always memcpy'd by its
 * producer before, and read by its consumer after, the matching atomic
 * hand-off, so the FIFO itself has no data race.
 *
 * Anything outside that shape needs the caller's own external serialization:
 *   - a second render-side or second capture-side thread (two producers or
 *     two consumers): fifo_write and the render/capture stats counters assume
 *     a single writer per side;
 *   - aec_reset() / aec_destroy() while either thread might still call in:
 *     both do plain (non-atomic) field writes and assume the streaming pair
 *     is quiesced first;
 *   - mixing aec_process_capture() with the offline aec_process() on the same
 *     instance concurrently with the streaming pair.
 *
 * Lockstep equivalence: aec_analyze_render(ref) immediately followed by
 * aec_process_capture(mic, out) from a SINGLE thread (the degenerate,
 * always-supported case of the contract) yields byte-identical output to
 * aec_process(mic, ref, out): the FIFO is pass-through and no event fires.
 *
 * Underrun (capture with empty FIFO): processed with a silent render hop
 * (aec_process_capture() substitutes the immutable fifo_zero_ref) and
 * AEC_BUF_RENDER_UNDERRUN returned. Overrun (render past FIFO capacity): the
 * incoming hop is discarded (drop-new); if the capture side finds the ring
 * full it skips to the freshest hop (catch-up) and reports
 * AEC_BUF_RENDER_OVERRUN. */
AecBufferingEvent aec_analyze_render(Aec* a, const float* ref);
AecBufferingEvent aec_process_capture(Aec* a, const float* mic, float* out);
/* Last buffering event from the most recent aec_process_capture(). */
AecBufferingEvent aec_last_buffering_event(const Aec* a);

int  aec_hop_size(const Aec* a);

/* Linear-AEC + external-RES integration hook (research / NN surface). */
typedef struct AecResContext {
    int            n_freqs;
    int            hop_size;
    const float*   linear_hop;       /* refined PBFDKF hop, before selection */
    const float*   formed_hop;       /* current hop underlying error_spec    */
    const Complex* echo_spec;
    const Complex* far_spec;
    const Complex* near_spec;
    /* Freq-domain seam for an external post-NR residual suppressor (the Python
     * AecResContext freq fields). Valid when enable_res=1 or
     * return_res_context=1; NULL otherwise. error_spec is the exact 50%-overlap
     * periodic-sqrt-Hann STFT of the formed linear output E(f), in AUDIO scale.
     * echo_spec is the matching formed echo estimate and
     * near_spec = error_spec + echo_spec the matching windowed capture
     * spectrum. res_gain is the SuppressionGain G_res(f), amplitude [0..1].
     * r2 = residual-echo PSD R^2(f) and comfort_noise = CNG N^2(f), BOTH
     * int16^2-scaled (divide by 32768^2 to reach the |E|^2 audio-power scale,
     * as run_res does). All length n_freqs. Under spatial_linear_context ==
     * AEC_CONTEXT_LINEAR_ONLY, res_gain, r2 and comfort_noise are NULL and
     * res_floor_protect is 0.
     * The pointers alias the AEC's internal per-hop buffers: read them before
     * the next processing call on this instance by ANY entry point
     * (aec_process(), aec_process_context(), aec_process_context_shared_far(),
     * aec_process_context_shared_far_from() or aec_process_capture()), since
     * each re-runs aec_process_core() and overwrites them in place. */
    const Complex* error_spec;     /* reconstructing WOLA linear E(f), audio scale */
    const float*   res_gain;       /* G_res(f), amplitude [0..1]; NULL when
                                     * AecConfig.spatial_linear_context is set
                                     * (this lane's G_res is never computed, and
                                     * its zero-initialised buffer would misread
                                     * as "fully suppressed")                 */
    const float*   r2;             /* residual-echo PSD R²(f), int16²-scaled   */
    const float*   comfort_noise;  /* CNG N²(f), int16²-scaled                 */
    float          far_power;
    float          erle_factor;
    float          dt_indicator;
    float          divergence;
    float          saturation_level;
    float          erl_estimate;
    float          shadow_dt;
    int            is_stationary_dt;
    int            filter_converged;
    int            filter_once_converged;
    int            epc_active;
    int            res_floor_protect;  /* this hop's near-recent-latch DT floor
                                        * decision; a fused consumer ORs the
                                        * contributing lanes' flags */
    int            usable_linear;      /* this hop's usable-linear-estimate
                                        * verdict. 0: r2 is the conservative
                                        * nonlinear estimate (far-end power
                                        * times a default echo-path gain), not
                                        * one derived from the linear echo
                                        * estimate -- it can stand well above a
                                        * residual that is not there at all.
                                        * Also 0 when the post block did not
                                        * run (r2 is NULL then) */
} AecResContext;

void aec_get_res_context(const Aec* a, AecResContext* ctx);

/* ── Linear-AEC / NN post-filter seam ─────────────────────────────────────
 * Minimal read-only view for an external neural post-filter (RES+NR) that
 * consumes the formed linear error and the SAME time-domain aligned far the
 * PBFDKF consumed this hop. Contract:
 *   - aligned_far_hop aliases the internal per-hop buffer (a->far_hop): no
 *     heap, no copy; valid only until the next process/reset call.
 *   - aligned_far_hop is byte-identical to what the linear filter read this
 *     hop. delay_state tells whether it was actually delay-compensated:
 *     UNLOCKED means the content is the RAW far (pre-acquisition, ring not yet
 *     filled, or delay estimation disabled); a consumer must not feed it to a
 *     small-search-range aligner as if it were aligned.
 *   - generation increments on every ring-offset change (first acquisition,
 *     confirmed shift including the flagless soft-recovery paths, reset) and
 *     saturates instead of wrapping. Poll it to invalidate cross-hop caches
 *     (far feature rings, attention histories); differencing delay_samples
 *     alone misses transient A->B->A shifts between polls.
 *   - CHANGED is reported exactly on the hop whose processing bumped
 *     generation; the next hop reads LOCKED again.
 *   - Out-of-range bulk delay is NOT detectable here. With a single dominant
 *     out-of-range path the estimator may remain UNLOCKED, but an in-range
 *     early reflection can also produce a confident lock at the wrong delay,
 *     and the seam has no ground truth to tell that mis-lock apart. Products
 *     must choose n from a measured route-delay envelope and own the
 *     fail-open/recovery policy. The getter itself mutates no state.
 *
 * PER-MODE SEMANTICS (cfg.delay_mode):
 *
 *   AEC_DELAY_MATCHED
 *     delay_samples is the applied ring offset (-1 before the first
 *     acquisition), delay_confidence is the estimator's own 0.0/0.5/1.0, and
 *     generation bumps at every ring-offset change.
 *
 *   AEC_DELAY_FIXED
 *     delay_samples == cfg.fixed_delay_samples, always (never -1).
 *     delay_confidence == 1.0: the delay was MEASURED out of band, so it
 *     carries no estimator uncertainty. generation NEVER moves while
 *     processing (no estimator re-locks); the only bump is aec_reset(), which
 *     invalidates the ring content a consumer may have cached. delay_state is
 *     LOCKED from the hop the ring first holds fixed_delay_samples + hop_size
 *     samples, i.e. UNLOCKED for the first ceil(fixed_delay_samples / hop_size)
 *     hops; the following process call, after writing the current hop, already
 *     reports LOCKED and serves the requested offset. Reporting LOCKED over
 *     raw far would break this seam's core promise ("UNLOCKED means the content
 *     is the RAW far") for the ~fixed_delay_samples of audio a
 *     small-search-range consumer is most likely to mis-handle. Steady state is
 *     LOCKED.
 *
 *   AEC_DELAY_EXTERNAL_ALIGNED
 *     aligned_far_hop IS the caller's own far hop (no ring exists; the only
 *     transform it can have seen is the optional reference soft-clip).
 *     delay_samples == 0 and delay_state == LOCKED, because the caller has
 *     CONTRACTED that far is already aligned: this is the one mode where
 *     "aligned" is an input, not a measurement. delay_confidence == 1.0, and
 *     generation never moves while PROCESSING (no estimator to re-lock), but
 *     aec_apply_external_realign() bumps it out of band on both its warm and
 *     reset outcomes: that call tells this instance its contracted alignment
 *     just changed, the event on which a cached far ring must be invalidated.
 *     The explicit mode is what makes this stronger statement legitimate; the
 *     deprecated enable_delay_est=0 shape could not tell "caller pre-aligned"
 *     from "estimator off and nobody aligned anything". */
typedef enum AecLinearDelayState {
    AEC_LINEAR_DELAY_UNLOCKED = 0,  /* far_hop is raw / not compensated */
    AEC_LINEAR_DELAY_LOCKED   = 1,  /* compensated with a stable offset */
    AEC_LINEAR_DELAY_CHANGED  = 2   /* compensated, but the offset changed
                                     * during THIS hop (flush far caches) */
} AecLinearDelayState;

typedef struct AecLinearContext {
    int hop_size;
    const float* formed_linear_hop;  /* formed linear error e_form when the
                                      * post-filter chain runs this config
                                      * (enable_res || return_res_context),
                                      * else the raw linear output          */
    const float* aligned_far_hop;    /* aliases a->far_hop (see contract)   */
    int   delay_samples;             /* applied ring offset; -1 = none      */
    float delay_confidence;          /* 0.0 / 0.5 / 1.0                     */
    AecLinearDelayState delay_state;
    unsigned int generation;
} AecLinearContext;

void aec_get_linear_context(const Aec* a, AecLinearContext* ctx);

/* ── Runtime debug/status introspection ───────────────────────────────────
 * Read-only status query for an integrator to poll (e.g. once per second) to
 * tell whether the pipeline is behaving on an embedded target. Fills a plain
 * struct from fields the engine already maintains: no logging inside the
 * library, no extra per-frame state, no added hot-path cost. The caller
 * (e.g. example/aec_wav.c --debug) prints. */
typedef struct AecDebugStatus {
    /* delay estimation (EchoPathDelayEstimator / matched-filter) */
    int    delay_samples;      /* current applied delay, samples; -1 = not yet
                                 * acquired (mirrors a->current_delay)         */
    float  delay_confidence;   /* MATCHED: the estimator's own 0.0/0.5/1.0
                                 * (delay_aec_confidence). FIXED and
                                 * EXTERNAL_ALIGNED: 1.0, since the alignment is
                                 * a caller contract (a bring-up measurement /
                                 * pre-aligned ref), not something this library
                                 * estimated. Same value as
                                 * AecLinearContext.delay_confidence.         */
    int    delay_updates;      /* matched-filter update count
                                 * (delay_aec_n_updates); 0 outside MATCHED */

    /* linear filter health */
    float  erle_windowed_db;   /* windowed ERLE (dB) from the last hop that
                                 * ran the post-filter chain (a->last_erle_windowed),
                                 * 0.0 before the post chain has run once      */
    int    usable_linear;      /* usable-linear-estimate gate state
                                 * (aec_state_usable_linear_estimate)          */
    int    filter_converged;   /* FilterConvergenceAnalyzer latch state
                                 * (a->convergence.converged, same value
                                 * AecResContext.filter_converged exposes)     */

    /* levels: EMA (alpha=0.95), NOT an instantaneous last-hop mean. They reuse
     * the power-tracking EMAs the ERLE machinery updates every hop anyway, so
     * reading them adds no per-frame cost. */
    float  near_power;         /* EMA of mic/near-hop power (a->near_power)    */
    float  out_power;          /* EMA of the linear, pre-RES-gain output power
                                 * (a->raw_error_power)                        */

    /* duty-cycle engagement census, cumulative since aec_create / aec_init /
     * aec_reset; duty_hops_run / duty_hops_total is the MEASURED matched-filter
     * engagement (duty_hops_* doc on the Aec struct above). Both are 0 whenever
     * no matched filter is constructed (delay_mode != AEC_DELAY_MATCHED), since
     * the duty machine then has no cost to have saved. */
    unsigned long long duty_hops_total;
    unsigned long long duty_hops_run;
} AecDebugStatus;

void aec_debug_status(const Aec* a, AecDebugStatus* out);

/* ── Static-pool memory breakdown ─────────────────────────────────────────
 * aec_get_mem_size() returns one total; this labels two subsets of that same
 * total so a caller (e.g. a CLI's --print-mem-size) need not re-derive them
 * from the config, which would be a second, driftable size computation.
 * total_bytes is always aec_get_mem_size(cfg) itself, called from inside this
 * function, so the two cannot disagree.
 *
 *   estimator_bytes  delay_aec_get_mem_size() for the resolved
 *                    (sample_rate, hop_size, delay_num_filters) triple;
 *                    0 outside AEC_DELAY_MATCHED (no estimator is built).
 *   ring_bytes       the alignment-ring float buffer, mode-aware (see
 *                    aec_ref_ring_samples() in aec.c / the manual's
 *                    ring-size-formula table); 0 for AEC_DELAY_EXTERNAL_ALIGNED
 *                    (no ring at all).
 *
 * Returns 0 (and zeroes *out) under exactly the condition aec_get_mem_size()
 * reports failure on: cfg is NULL, or aec_validate_config(cfg) rejects it. */
typedef struct AecMemBreakdown {
    size_t total_bytes;
    size_t estimator_bytes;
    size_t ring_bytes;
} AecMemBreakdown;

int aec_get_mem_breakdown(const AecConfig* cfg, AecMemBreakdown* out);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_H */
