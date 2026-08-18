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

#include "fft_wrapper.h"
#include "hpf.h"
#include "saturation.h"
#include "delay_aec3.h"
#include "detectors.h"
#include "pbfdkf.h"
#include "epc_shadow.h"
#include "render_signal_analyzer.h"
#include "aec3_post.h"   /* Aec3Post + Aec3PostRunObj/In/Scratch + sub-modules */

typedef enum { AEC_PRESET_MILD = 0, AEC_PRESET_BALANCED, AEC_PRESET_AGGRESSIVE } AecPreset;

/* ── Delay mode (init-time, immutable for the instance's lifetime) ────────
 * The single source of truth for how the far-end is aligned to the mic.
 * Mirrors Python modules/enums.py AecDelayMode (identical integer values).
 *
 *   AEC_DELAY_MATCHED           the AEC3 matched-filter bank estimates the
 *                               delay online and a reference ring applies it.
 *                               Requires delay_num_filters in [1, 5] and
 *                               fixed_delay_samples < 0.
 *   AEC_DELAY_FIXED             no estimator; the caller has MEASURED the
 *                               system delay (bring-up) and the reference
 *                               ring applies exactly fixed_delay_samples.
 *                               Requires fixed_delay_samples >= 0.
 *   AEC_DELAY_EXTERNAL_ALIGNED  no estimator and no ring; the caller
 *                               guarantees the `ref` it passes is ALREADY
 *                               aligned to `mic`. Requires
 *                               fixed_delay_samples < 0.
 *
 * Illegal mode/field combinations are REJECTED (aec_validate_config), never
 * silently normalised -- a field that the selected mode cannot honour is a
 * caller mistake, not something to ignore. See aec_config_resolve_delay()
 * for the deprecated-field translation layer. */
typedef enum AecDelayMode {
    AEC_DELAY_MATCHED = 0,
    AEC_DELAY_FIXED = 1,
    AEC_DELAY_EXTERNAL_ALIGNED = 2
} AecDelayMode;

typedef struct AecConfig {
    int    sample_rate;        /* 16000 */
    int    fft_size;           /* frame size; 256/512 @16k, 1024 @48k */
    int    filter_length;      /* 832 samples (52 ms) */
    int    n_partitions;       /* derived if 0 */
    float  mu;                 /* 0.3 */
    float  delta;              /* 1e-8 */

    /* toggles */
    int    enable_cng;             /* 1 */
    /* DEPRECATED translation-layer mirror of `delay_mode` (below). Kept so
     * pre-delay_mode callers keep compiling and behaving identically; it is
     * NOT a second source of truth. aec_config_resolve_delay() folds it into
     * delay_mode once, before validation, and every downstream read in this
     * library is of delay_mode alone. Scheduled for removal once all callers
     * have moved to delay_mode. See aec_config_resolve_delay() for the exact
     * mapping table and the conflict rule. */
    int    enable_delay_est;       /* 1 (deprecated; see delay_mode) */
    int    enable_highpass;        /* 1 (mic only) */
    int    enable_saturation;      /* 1 */
    int    enable_shadow;          /* 1 */
    int    enable_res;             /* 1 */
    int    saturation_softclip_ref;/* 1 */
    /* Linear-AEC + external-RES seam (mirrors Python AecConfig.return_res_context,
     * config.py:332). When set, the AEC3 post block RUNS (computes res_gain / R² /
     * comfort_noise / windowed error spectrum and exposes them via AecResContext)
     * but, if enable_res==0, its suppression is NOT applied — the time output stays
     * the linear residual (raw_output). Lets a caller run AEC(linear) → NR → RES
     * externally. Default 0 (memset) → production cascade untouched. */
    int    return_res_context;     /* 0 */
    /* This lane's own AEC3 SuppressionGain output (G_res) is never computed
     * -- a downstream fused/beamformed post-stage (e.g. a multi-channel
     * wrapper) recomputes an equivalent gain from combined multi-lane data
     * instead. The DominantNearend hold-state (which the next hop's ERLE
     * onset decision, and therefore r2, depend on) is still updated every
     * hop. comfort_noise is computed independently (it is one of DNE's own
     * inputs, not something DNE affects) and is unaffected either way.
     * Only valid when enable_res==0 && return_res_context==1 (the
     * context-only seam); aec_validate_config() rejects any other
     * combination, since apply_output() -- the only consumer of the skipped
     * gain -- only runs when that seam is NOT active. Default 0 (memset). */
    int    spatial_linear_context; /* 0 */

    /* scalar tunables (balanced base). shadow_err_alpha/warmup_frames/
     * epc_hangover are wall-clock-authored at the legacy hop=160/sr=16000
     * (10 ms) grid and retimed live for the real grid in aec_carve() --
     * construction time, the one place guaranteed to see the final
     * resolved hop after any caller override (e.g. aec_wav.c's
     * --fft-size), not aec_config_defaults(). Values below are reference-grid
     * figures and may differ from the effective values stored by an Aec. */
    float  shadow_err_alpha;       /* 0.80 @ 10 ms hop */
    float  shadow_mu_min;          /* 0.5  */
    float  shadow_mu_nlms;         /* 0.5  */
    int    warmup_frames;          /* 100 hops = 1000 ms @ 10 ms hop */
    int    epc_hangover;           /* 20 hops = 200 ms @ 10 ms hop */
    float  epc_total_rise;         /* 1.5  */
    float  epc_delta_threshold;    /* 0.3  */
    float  epc_mu_floor;           /* 0.5  */
    /* MATCHED-only: together these size the reference SEARCH ring
     * (max(delay_buffer_ms, max_delay_ms + 4096 samples)). Inert under
     * FIXED (sized from fixed_delay_samples + hop) and EXTERNAL_ALIGNED
     * (no ring at all) -- see aec_ref_ring_samples() in aec.c. */
    float  max_delay_ms;           /* 1024 */
    float  delay_buffer_ms;        /* 2048 */
    float  delay_est_init_s;       /* 0.3  */
    float  delay_est_period_s;     /* 0.5  */
    /* Matched-filter bank size, [1, DA_NUM_FILTERS]; mirrors Python
     * AecConfig.delay_num_filters. 5 = the AEC3 default geometry and the
     * only value bench/dataset runs may use. Smaller is a COMPUTE AND
     * MEMORY knob for the embedded target, where the bulk system delay is
     * already compensated out-of-band and the matched filter only has to
     * track the residual: the reliable reach drops to 125/221/317/413/509 ms
     * for n=1..5, each dropped filter removes ~4.2 MMAC/s of full-rate
     * search, AND aec_get_mem_size() drops by exactly 5,728 B per filter on
     * every grid (the bank, the render ring and both lag histograms are
     * pool-carved to this n -- see delay_aec3.h's MEMORY CONTRACT block).
     * This is therefore part of the POOL LAYOUT: init-time immutable like
     * the signal grid, and changing it means re-querying aec_get_mem_size()
     * and re-initialising.
     * Out-of-range values are REJECTED by aec_validate_config (they are a
     * caller mistake, not something to silently normalise), and so is any
     * value other than the default when delay_mode is not MATCHED (there is
     * no matched filter to size, so a non-default n would be a silently
     * ignored input -- see delay_mode / fixed_delay_samples at the end of
     * this struct). */
    int    delay_num_filters;      /* 5    */
    float  highpass_cutoff_hz;     /* 80   */
    float  saturation_threshold;   /* 0.95 */
    float  kalman_q_high;          /* 1e-3 */
    float  kalman_q_low;           /* 1e-6 */
    int    delay_acquire_protect_converged;   /* 1 */
    /* Warm tap-transfer on first delay acquisition (v3.24.1, default ON):
     * shift the learned IR by the acquired delay instead of zeroing, so the
     * cold-start cancellation survives the realign (the "1.14s vertical line"
     * fix). Gated on inst-ERLE peak > _inst_erle_db AND delay < tap reach.
     * _protect_inst_erle is the A/B-rejected option-A (block the realign). */
    int    delay_acquire_warm_transfer;       /* 1 */
    float  delay_acquire_inst_erle_db;        /* 4.0 */
    int    delay_acquire_protect_inst_erle;   /* 0 */
    /* Backward-jump quarantine on a delay CHANGE (Path B). Sibling of
     * delay_acquire_protect_converged above -- which guards the FIRST
     * acquisition only and is untouched by these two fields. DEFAULT OFF, so
     * the shipped path is byte-identical to a build without the mechanism.
     *
     * WHAT IS QUARANTINED, and nothing else: a candidate strictly EARLIER
     * than the delay already in force (new_delay < current_delay -- the
     * backward / pre-echo direction) offered while the linear filter is still
     * demonstrably cancelling at the applied alignment
     * (aec_linear_is_cancelling()). FORWARD jumps -- a LARGER delay, which
     * pre-echo mis-attribution cannot produce -- are never quarantined, and
     * neither is the first acquisition.
     *
     * TIME-BOUNDED, never a veto. The countdown is armed on the first refusal
     * and spends one tick per estimation cycle; after
     * delay_backward_quarantine_s worth of cycles the candidate is ACCEPTED.
     * The window is converted to cycles once, in aec_carve(), against the
     * RESOLVED hop and sample rate (minimum 1, so a sub-hop window still
     * quarantines for exactly one cycle instead of silently disabling the
     * mechanism). Two ways out early:
     *   (a) cancellation collapses -- common for a hard path replacement,
     *       but not guaranteed for multipath/path-addition changes -- and the
     *       candidate is accepted on that cycle;
     *   (b) nothing else. The estimator publishes no dominance signal beyond
     *       persistence: delay_aec3_confidence() is histogram consistency
     *       quantised to 0 / 0.5 / 1.0 and carries no quality information
     *       (the pre-echo candidate reaches REFINED too), and Path B already
     *       gates on it. "The candidate kept being offered" IS the
     *       persistence evidence, and the window is what measures it.
     *
     * So a mis-lock is DELAYED by at most the window, never cured. Curing it
     * is estimator work, out of scope for this guard.
     *
     * History (one line, so it is not re-derived): the first version of this
     * mechanism refused ALL differing candidates while cancelling, with no
     * bound. In multipath / path-addition scenes the surviving old reflection
     * keeps ERLE up, so a genuine new path was vetoed indefinitely -- measured
     * at old/new gains 0.4/0.5 with no re-lock through hop 850, and 0.5/0.5
     * never. Candidate direction + a bound are what replaced it. */
    int    delay_backward_quarantine_enabled; /* 0 */
    float  delay_backward_quarantine_s;       /* 1.0 */

    /* DT-deg recovery stack (default ON, mirrors Python 16285fd). The
     * energy-based held near-end gate (_ne_recent_frames) arms two levers
     * during double-talk only (FS never raises dt_from_energy): */
    int    dt_aware_recovery_soft;            /* 1 — soft (non-destructive)
                                               * Path A/B/EPV/shadow_rise realign
                                               * while near recently present */
    int    dt_aware_res_floor_enabled;        /* 1 — DT-gated RES min-gain floor */
    float  min_gain_floor_dt_db;              /* -16.0 — DT floor (dB) */
    /* ne_recent gate parameters (mirror AecConfig.ne_recent_*). threshold is
     * float32-by-design (Python bit-exact parity retired; f32/f64 drift
     * across this threshold is accepted). ne_recent_hold is wall-clock-
     * authored @ 10 ms hop and retimed live in aec_carve();
     * ne_recent_sustain is a genuine consecutive-event count
     * (NOT a duration) and is intentionally left unretimed. */
    float  ne_recent_threshold;               /* 0.3 */
    int    ne_recent_hold;                    /* 150 hops = 1500 ms @ 10 ms hop */
    int    ne_recent_sustain;                 /* 3 (event count, not retimed) */

    /* preset strength axis (the ONLY field mild/balanced/aggressive differ
     * in): SuppressionGain far-active min-gain floor in dB
     * (mild -20 / balanced -28 / aggressive -38). */
    float  min_gain_floor_far_active_db;

    /* FilterMisadjustmentEstimator (AEC3 ScaleFilter). stable/hangover_frames
     * are wall-clock-authored @ 10 ms hop and retimed live in aec_carve(). */
    int    filter_misadjustment_stable_frames;   /* 30 hops = 300 ms @ 10 ms hop */
    int    filter_misadjustment_hangover_frames;  /* 100 hops = 1000 ms @ 10 ms hop */
    float  filter_misadjustment_scale_min;        /* 0.5 */
    float  filter_misadjustment_scale_max;        /* 2.0 */

    /* ── delay mode (appended at the end deliberately: every field above
     * keeps the byte offset it had before delay_mode existed) ───────────── */
    /* The single source of truth for far/mic alignment; see AecDelayMode.
     * init-time immutable -- changing it requires destroy/re-init, because
     * the pool layout differs per mode. Default AEC_DELAY_MATCHED. */
    AecDelayMode delay_mode;       /* AEC_DELAY_MATCHED */
    /* FIXED only: the measured system delay in NATIVE-rate samples, >= 0.
     * Must be < 0 (the -1 "unset" sentinel) in MATCHED and
     * EXTERNAL_ALIGNED -- a fixed delay those modes cannot honour is
     * rejected, not ignored. Mirrors Python AecConfig.fixed_delay_samples.
     * Sizes the reference ring exactly: FIXED carves
     * `fixed_delay_samples + hop_size` samples and ignores max_delay_ms /
     * delay_buffer_ms entirely (those size the MATCHED SEARCH ring, and
     * there is no search here). Same rule as the Python orchestrator's
     * ref_ring_samples(). */
    int    fixed_delay_samples;    /* -1 */
} AecConfig;

void aec_config_defaults(AecConfig* cfg, int sample_rate);
void aec_config_from_preset(AecConfig* cfg, AecPreset preset, int sample_rate);

/* ── Deprecated-field translation layer (SHORT-TERM compatibility) ────────
 * Folds the deprecated `enable_delay_est` mirror into `delay_mode`, in
 * place, so that afterwards `delay_mode` is the ONLY field that decides
 * alignment behaviour. Idempotent; called automatically by
 * aec_validate_config / aec_get_mem_size / aec_create / aec_init, so a
 * caller never has to invoke it -- it is public only so tests and
 * integrators can inspect the resolved mode without constructing an Aec.
 *
 * Mapping (the ONLY three legacy shapes that ever existed):
 *
 *   enable_delay_est | fixed_delay_samples | delay_mode in | resolved mode
 *   -----------------+---------------------+---------------+---------------
 *    1 (default)     | any                 | any           | delay_mode (unchanged)
 *    0               | >= 0                | MATCHED (dflt)| FIXED
 *    0               | <  0                | MATCHED (dflt)| EXTERNAL_ALIGNED
 *    0               | any                 | FIXED         | FIXED
 *    0               | any                 | EXTERNAL_ALIGN| EXTERNAL_ALIGNED
 *
 * i.e. `enable_delay_est == 1` (its default) carries no information and
 * leaves delay_mode alone; clearing it is the legacy way of saying "not
 * MATCHED", and it only translates while delay_mode is still at its own
 * default. A caller that sets BOTH already agrees with itself, since the
 * only non-default delay_mode values are exactly the two non-MATCHED
 * outcomes the legacy field can select.
 *
 * On return `enable_delay_est` is rewritten to the mirror of the resolved
 * mode (1 for MATCHED, 0 otherwise) so a config round-tripped through this
 * helper is self-consistent whichever field a caller reads.
 *
 * Returns 0 on success, -1 if cfg is NULL or delay_mode is not one of the
 * three enumerators. */
int aec_config_resolve_delay(AecConfig* cfg);

/* ── Signal grid: ONE resolver, shared by every entry point ───────────────
 * (sample_rate, fft_size) is the complete init-time description of the
 * frame geometry; everything else is derived. This project fixes
 * frame_size == fft_size and hop_size == fft_size/2, but that derivation
 * lives in exactly ONE place -- aec_resolve_signal_grid() -- and
 * aec_validate_config(), aec_get_mem_size(), aec_create() and aec_init()
 * all go through it, so a caller can never be sized against one grid and
 * initialised against another.
 *
 * fft_size == 0 ("auto") is REJECTED here: 16 kHz has TWO production grids
 * (256 and 512) and a library that guesses would silently pick one. The
 * convenience factories aec_config_defaults() / aec_config_from_preset()
 * fill a concrete fft_size for the caller (16k -> 256, 48k -> 1024,
 * 8k -> 256), but they are the ONLY place a default is applied -- the core
 * never re-guesses afterwards. Likewise a CLI may pick a product default
 * for a user who did not specify one, but must resolve it to a nonzero
 * fft_size before calling in.
 *
 * Supported grids:
 *
 *   sample rate | fft/frame | hop | status
 *   ------------+-----------+-----+------------------------------------
 *    16000      |  256      | 128 | product (default at 16 kHz)
 *    16000      |  512      | 256 | product (alternate, higher compute)
 *    48000      | 1024      | 512 | product
 *     8000      |  256      | 128 | LEGACY -- supported, NOT a product
 *                                 | grid; kept for the existing 8 kHz
 *                                 | E2E-parity/structural tests and the
 *                                 | Audio_ALG MONO pipeline. The 4-channel
 *                                 | pipeline contracts to the three product
 *                                 | grids only. is_legacy is set for it. */
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

/* Sample-rate whitelist query (F05: no sample-rate validation anywhere).
 * "Is there at least one supported grid at this rate" -- a read of the SAME
 * grid table aec_resolve_signal_grid() uses, for callers (e.g.
 * example/aec_wav.c) that need to reject an unsupported input WAV rate
 * before they have chosen an fft_size. Returns 1 for 8000/16000/48000.
 * NOTE: a true sample_rate here does NOT mean any fft_size will do -- the
 * pair still has to resolve. */
int aec_is_valid_sample_rate(int sample_rate);

/* The product-default fft_size for a rate, or 0 if the rate is unsupported.
 * The single implementation behind aec_config_defaults()'s convenience
 * fill; exposed so a CLI can resolve a user's unspecified grid to a
 * concrete value before calling the core, instead of hard-coding its own
 * copy of the mapping. */
int aec_default_fft_size(int sample_rate);

/* ── opaque-ish context ────────────────────────────────────────────────── */
/* Streaming render/capture buffering events (mirror AEC3
 * render_delay_buffer.h:28 BufferingEvent). Returned by the streaming API. */
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

    /* delay (AEC3 matched filter) + ring.
     * `delay` is a METADATA struct: its arrays (bank, render ring, both lag
     * histograms, 48 kHz sidechain scratch) live in this instance's pool,
     * carved by aec_carve() at the size aec_get_mem_size() budgeted for the
     * resolved (sample_rate, hop, delay_num_filters) triple. Nothing here is
     * carved at a compile-time maximum. Only MATCHED carves it at all --
     * when has_delay == 0 every pointer inside is NULL and every size 0.
     * has_delay == 1 iff cfg.delay_mode == AEC_DELAY_MATCHED, i.e. iff the
     * matched-filter ESTIMATOR was constructed (mirrors Python's
     * `delay_est is not None`). The reference RING is a separate question:
     * it exists for MATCHED and FIXED alike and only EXTERNAL_ALIGNED has
     * none (mirrors Python's `_delay_active`). Ask aec_ring_active(&cfg)
     * (aec.c) for "is there a ring" -- never has_delay, and never a
     * re-spelled delay_mode comparison, so the two questions cannot silently
     * merge again. */
    DelayAec3 delay;       int has_delay;
    float* ref_ring;       int ref_ring_size, ref_ring_write, ref_ring_filled;
    int    current_delay;  /* MATCHED: -1 until first acquisition.
                            * FIXED:   cfg.fixed_delay_samples, always.
                            * EXTERNAL_ALIGNED: 0 (nothing to apply). */
    int    pending_delay;  int has_pending; int pending_delay_ttl;
    /* Backward-jump quarantine (cfg.delay_backward_quarantine_*).
     * _left: estimation cycles still to spend before a refused EARLIER
     *        candidate is accepted anyway. -1 = DISARMED, which is the only
     *        state that may re-arm; 0 = armed and EXPIRED, i.e. this cycle
     *        accepts. Disarming happens on any cycle the guard is not
     *        engaged (no backward candidate, cancellation collapsed, or the
     *        feature is off), which is also what disarms it after an
     *        acceptance -- once current_delay moves the candidate is no
     *        longer a backward jump.
     * _hops: the window in cycles, derived once in aec_carve(); >= 1. */
    int    delay_quarantine_left;
    int    delay_quarantine_hops;
    /* Alignment-generation token for external consumers (AecLinearContext).
     * Bumped (saturating, never wrapping) at EVERY write that changes the
     * alignment the filter is matched to -- first acquisition, confirmed shift
     * (soft AND hard recovery paths share the same write site), aec_reset(),
     * and aec_apply_external_realign() on both its outcomes (the one bump
     * EXTERNAL_ALIGNED has, since it owns no ring offset to move). External
     * pollers cannot reconstruct this from delay_samples differences alone:
     * the soft-recovery paths deliberately set no other flag, and in
     * EXTERNAL_ALIGNED delay_samples is 0 by contract. */
    unsigned int delay_generation;
    unsigned int delay_gen_hop_start;  /* snapshot at hop entry (CHANGED detect) */
    int    far_hop_aligned;    /* 1 = this hop's far_hop was ring-compensated
                                * (or the applied delay is exactly 0); 0 = raw,
                                * unaligned far (pre-acquisition / ring not yet
                                * filled / delay estimation disabled) */
    /* Delay-estimator duty-cycle state — ALWAYS ACTIVE (baked in, no config
     * gate): analyses every hop until the delay estimate is solid (confidence
     * 1.0) AND unchanged for delay_est_init_s seconds, then self-gates the
     * matched-filter analysis down to 1 hop in every K, where K = max(2,
     * round(delay_est_period_s/hop_s)/5) — 10 with the default 0.5 s period
     * @10 ms hop, cutting ~90% of the matched-filter cost. The estimator's
     * decimators + render ring stay FED every hop regardless (only the
     * analysis decimates), so the matched filter never sees gapped audio.
     * Full-rate analysis resumes immediately when (a) the estimate changes or
     * loses solidity, or (b) the ERLE watchdog fires: erle_windowed drops
     * >6 dB below its running peak (peak leaks ~0.1 dB/s; armed only once the
     * peak exceeds 6 dB, so it cannot fire before first convergence).
     * Sampled 60-case AECMOS cost: <=+0.014 / worst -0.006 — zero-cost. */
    int    duty_active;        /* 1 = decimated analysis in effect */
    int    duty_stable_hops;   /* consecutive solid+unchanged hops (arming) */
    int    duty_pos;           /* position in the 1-in-K analysis cycle */
    int    duty_last_delay;    /* estimate on the previous hop (change detect) */
    float  duty_erle_peak;     /* leaky running peak of erle_windowed (dB) */
    /* Engagement census (productization plan §7 measurement method): the
     * doc comment above claims the duty machine cuts "~90% of the
     * matched-filter cost" as a DESIGN figure, which only materialises on
     * hops the machine actually decimates -- on an echo path whose estimate
     * keeps moving it re-arms every time and never decimates at all. These
     * two counters make the REALISED saving measurable per run instead of
     * assumed: duty_hops_run / duty_hops_total is the true matched-filter
     * engagement, and 1 minus that ratio the true saving. Incremented at the
     * single call site that consumes run_filter (aec_process_core), so the
     * census cannot drift from what was actually executed. uint64 because a
     * long-running stream at a 128-sample hop passes 2^31 hops in ~2 years
     * of continuous audio. Diagnostic only -- nothing in the library reads
     * them back; exposed read-only via AecDebugStatus and printed by
     * example/aec_wav.c. Cumulative since aec_create/aec_init/aec_reset. */
    unsigned long long duty_hops_total;  /* hops through the delay block   */
    unsigned long long duty_hops_run;    /* of those, hops that actually
                                          * ran the matched filter
                                          * (run_filter == 1)               */

    /* linear filters */
    PBFDKF main_filter;
    PBFDAF shadow_filter;  int has_shadow;

    /* detectors / EPC / regime / RSA */
    RenderActivity    render_activity;
    /* De-stacked RenderActivity scratch (formerly a fixed `float[1024]`
     * local inside render_activity_update()); pool-carved, sized
     * ceil(hop_size/8) -- see detectors.h. */
    float*            ra_pairwise_scratch;
    FilterConvergence convergence;
    DoubleTalk        dt_analyzer;
    EpcDetector       epc;
    ShadowCopy        regime;
    RenderSignalAnalyzer rsa;   int64_t* rsa_counters;

    /* AEC3 post chain (driver + sub-module objects, all caller-owned storage) */
    FftHandle*  post_fft;
    Aec3Post    post;
    AecState              a3_state;   AecStateStorage a3_state_st;
    ResidualEchoEstimator a3_ree;
    SuppressionGain       a3_sg;
    StationarityEstimator a3_stat;
    LinearFilterSelect    a3_lfs;
    Aec3PostRunScratch    a3_sc;
    /* tuning arrays for SuppressionGain (pointers into baked const tables) */

    /* AEC3 pending EPV mirror + stationarity read-state */
    int    pending_gain_change;   /* bool */
    int    pending_delay_change;  /* -1 == None, else DelayAdjustment */
    int    stationarity_active_hops;
    int    stationarity_converge_hops;
    int    non_zero_render_seen;
    float  render_peak_floor;
    int    block_stationary_next;   /* _block_stationary_for_next_hop latch */

    /* per-frame scalars (mirror orchestrator instance fields). float32-by-
     * design (Stage-2 f32 conversion; Python f64 bit-exact parity retired,
     * f32/f64 drift accepted — see aec.c). */
    float  saturation_level;
    float  erl_estimate;
    float  main_err_smooth, shadow_err_smooth;
    long   shadow_frame_count;
    int    epc_render_forced_remaining;
    float  erle_window_near, erle_window_err;
    float  erle_factor_prev;
    float  inst_erle_smooth;
    /* inst-ERLE slope ring (mirrors Python orchestrator _erle_slope_buf, a
     * deque maxlen ~500ms/hop). Holds get_erle_instant() dB values; the warm
     * tap-transfer gate reads max() of the last 15. Filled every freq-path hop
     * OUTSIDE the enable_res gate (Python appends unconditionally). */
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
    long   frame_count;

    /* DT-deg recovery: held "near-end seen recently" gate (mirrors Python
     * _ne_above consecutive-above counter + _ne_recent_frames hold countdown). */
    int    ne_above;
    int    ne_recent_frames;

    /* poor-coarse rescue (process loop) */
    int    poor_coarse_counter;
    int    coarse_reset_hangover;
    /* Track F: sustained leakage-diverged gate counter (clamped 0..
     * 2*leakage_div_sustain_hops). Mirrors
     * orchestrator._leakage_div_sustained_counter. */
    int    leakage_div_sustained_counter;
    /* Live-computed thresholds for the four counters above (+
     * stat_dt_hangover), all previously frozen at the legacy hop=160/
     * sr=16000 grid's hop-count result (2/10/5/80) instead of being
     * recomputed for the live hop_size/sample_rate -- see aec.c's
     * poor-coarse-rescue / Track-F / stationary-DT-hangover blocks.
     * Mirrors orchestrator.py's _aec3_scale.blocks_to_hops(5)/(25) and
     * ms_to_hops(50.0)/(800.0) calls at those same sites. */
    int    poor_coarse_threshold_hops;    /* was hardcoded 2 */
    int    coarse_reset_hangover_hops;    /* was hardcoded 10 */
    int    leakage_div_sustain_hops;      /* was hardcoded 5 (clamp was 2x = 10) */
    int    stat_dt_hangover_hops;         /* was hardcoded 80 */

    /* FilterMisadjustmentEstimator accumulators */
    float  misadj_e2_acum, misadj_y2_acum;
    int    misadj_n_acum;
    float  misadj_inv;
    int    misadj_overhang;
    int    misadj_stable_count;
    int    misadj_hangover_remaining;

    /* erle_windowed cache for the delay Path-A _already_cancelling guard
     * (= last frame's erle_windowed dB). */
    float  last_erle_windowed;

    /* RES-context stash (last frame) */
    float  last_far_power, last_shadow_dt, last_dt_indicator;
    /* Retimed ERL tracker EMAs (per hop, 10 ms authored grid). */
    float  alpha_erl_tracking, alpha_erl_converged;
    int    last_is_stationary_dt;

    /* hop scratch */
    float* near_hop;
    float* far_hop;
    float* raw_output;
    float* shadow_out;
    float* final_out;
    float* filter_taps_full;   /* [n_partitions × hop] */

    /* per-hop freq-bin scratch (Tier-1 stack-safety fix): these used to be
     * `float x[8192]` locals inside aec_process() / mean_sq(), sized far
     * beyond the actual n_freqs (K) they ever hold and with no assertion
     * tying 8192 to fft_size — a larger FFT config would silently smash the
     * stack, and some scopes stacked 2-3 of them at once (~96 KB worst case
     * in one block), which is unsafe headroom for an embedded RTOS task
     * stack (often 16-64 KB total). Moved into the Aec instance (malloc'd in
     * aec_create, pool-carved in aec_init, sized exactly K — or hop for
     * scr_sq, which mean_sq() only ever runs over hop-length buffers) so
     * stack usage no longer depends on n_freqs at all. Bit-exact: same
     * values, same order, only storage location changed. */
    float* scr_sq;          /* mean_sq() scratch, size hop_size (not K) */
    float* scr_e2_echo;
    float* scr_e2_near;
    float* scr_rsa_psd;
    float* scr_rsa_mask;
    float* scr_mu_buf_pre;
    float* scr_e2coa_pre;
    float* scr_mu_buf;
    float* scr_far_psd;
    float* scr_e2ref_arr;
    float* scr_e2coa_arr;
    float* scr_erl_arr;

    /* ── streaming render-hop FIFO (async render/capture decoupling) ──
     * Only exercised by the streaming API (aec_analyze_render /
     * aec_process_capture). The lockstep aec_process() bypasses it entirely,
     * so offline byte-exact parity with Python is untouched. In lockstep use
     * (one analyze_render then one process_capture) the FIFO is pass-through
     * (fifo_write one ahead of fifo_read, no event) → identical output to
     * aec_process(). */
    /* SPSC ring with one writer per cursor. On overrun the producer drops the
     * new hop; when a full ring separates the cursors, the consumer catches
     * up to the freshest hop. Cross-thread cursor access uses GCC/Clang
     * __atomic builtins on plain fields so the public layout remains C89-
     * compatible.
     * Per-field protocol:
     *   - fifo_write: producer-owned monotonic unsigned write sequence
     *     number (ever-increasing, indexed into the ring with `% cap` at
     *     each use — never stored pre-wrapped — so unsigned wraparound
     *     after a decades-long uptime is well-defined and harmless). Sole
     *     writer: the render thread, in aec_analyze_render(). Read by its
     *     own thread as a plain load (no other writer of this field exists,
     *     so that load cannot race); RELEASE-stored only after the new
     *     hop's memcpy into the ring slot has completed, so that store
     *     publishes the payload write to whichever thread's ACQUIRE load of
     *     fifo_write next observes the incremented value.
     *   - fifo_read: consumer-owned monotonic unsigned read sequence number,
     *     same wraparound treatment. Sole writer: the capture thread, in
     *     aec_process_capture() — including its own catch-up path (unlike
     *     the earlier design, the producer never touches fifo_read here).
     *     Plain-loaded by its own thread; RELEASE-stored only AFTER
     *     aec_process() has returned for the claimed hop (see the
     *     slot-lifetime note below), so that store both publishes the new
     *     read cursor and marks the slot free to the producer's next
     *     ACQUIRE load of fifo_read.
     *   - occupancy is always `fifo_write - fifo_read` (unsigned
     *     subtraction). fifo_cap_hops is forced to a power of two
     *     specifically so this stays correct across the 2^32 wrap of either
     *     cursor: `(x mod 2^32) mod cap == x mod cap` for every unsigned x
     *     only when cap divides 2^32 exactly (see aec_derive_dims in
     *     aec.c). The invariant `0 <= fifo_write - fifo_read <= cap` holds
     *     at every instant, by induction: the producer only advances
     *     fifo_write after observing `fifo_write - fifo_read_obs < cap` for
     *     some fifo_read_obs <= the true fifo_read at that moment (an
     *     ACQUIRE load can only lag, never lead, its writer), so the
     *     post-increment true occupancy is <= cap; the consumer only
     *     advances fifo_read after observing `fifo_write_obs - fifo_read >
     *     0` for some fifo_write_obs <= the true fifo_write, so the
     *     post-increment true occupancy is >= 0. Slot aliasing — two live
     *     claims resolving to the same physical ring index — would require
     *     fifo_write == fifo_read (mod cap), i.e. fifo_write - fifo_read
     *     equal to either 0 or cap; both are exactly the boundary values the
     *     guards above forbid crossing without the matching drop-new /
     *     catch-up event, so the producer never writes a slot the consumer
     *     might still be reading and the consumer never reads a slot the
     *     producer has not yet fully written.
     *   - The producer's occupancy check is staleness-safe in the
     *     conservative direction: because fifo_read_obs <= the true
     *     fifo_read, the producer's computed occupancy
     *     (fifo_write_true - fifo_read_obs) can only OVER-estimate the true
     *     occupancy — worst case it refuses to write (a spurious drop-new)
     *     when there was actually room; it can never UNDER-estimate and so
     *     never overwrites a slot the consumer has not released.
     *   - The ring slot's audio payload: the producer memcpy's the new hop
     *     into the ring BEFORE its RELEASE store of fifo_write (see above);
     *     the consumer only RELEASE-stores its advanced fifo_read AFTER
     *     aec_process() has returned — its first act, at the top of
     *     aec_process(), is to memcpy the passed-in `ref` OUT of the ring
     *     into a->far_hop — i.e. the slot is held "claimed" for the whole
     *     hop, not released the instant the copy-out is done. That is more
     *     conservative than strictly necessary (it can turn one extra
     *     producer call into a reported drop-new under sustained skew) but
     *     needs no change inside aec_process() itself.
     *   - fifo_count: RETIRED by this rewrite — always 0. Kept ONLY so every
     *     field below it keeps its pre-existing byte offset (layout
     *     stability). Nothing in aec.c reads or writes it except the
     *     zeroing at init/reset.
     *   - fifo_zero_ref: an immutable all-zero hop, carved from the pool at
     *     init and re-zeroed at reset — otherwise never written by either
     *     streaming thread. This is the underrun reference: unlike the
     *     earlier design (which memset a ring slot in place on underrun),
     *     the ring itself is never written by the capture thread, so
     *     aec_process_capture() has nothing to race against on this path.
     *   - render_call_count / capture_call_count / last_buffering_event:
     *     each is written by exactly one side (render_call_count only by
     *     the render thread, the other two only by the capture thread), so
     *     plain increments are race-free BY CONSTRUCTION and are left as
     *     plain `long`/`int` ops — no atomics needed for the writer side.
     *     last_buffering_event is additionally given a relaxed atomic store
     *     (write) / relaxed atomic load (aec_last_buffering_event() read)
     *     purely so a third thread polling it is never a formal data race;
     *     the SPSC contract does not promise it is fresher than "as of the
     *     last capture call this thread happens to observe".
     * aec_reset()/aec_create()/aec_init() write these fields as plain ints —
     * correct only because the contract requires the caller to fully
     * serialize with both the render and capture threads before resetting
     * or destroying a streaming-mode Aec (see the contract text below).
     * This design is TSan-clean with zero suppressions (see
     * test/test_fifo_spsc.c's header comment). */
    float* render_fifo;        /* [fifo_cap_hops × hop_size] ring of render hops */
    int    fifo_cap_hops;      /* capacity in hops — forced to a power of two, see aec_derive_dims */
    int    fifo_count;         /* RETIRED — always 0, kept only for layout stability, see above */
    int    fifo_read, fifo_write;  /* fifo_read: consumer-owned monotonic unsigned cursor; fifo_write: producer-owned monotonic unsigned cursor (see above) */
    long   render_call_count, capture_call_count;  /* single-writer-per-field stats, plain ops (see above) */
    int    last_buffering_event;   /* AecBufferingEvent of the last capture step — relaxed-atomic (see above) */

    int is_static;

    /* Single allocation owned by aec_create(); NULL for caller-pool aec_init(). */
    void*  heap_arena;

    /* Immutable all-zero reference used on streaming FIFO underrun. */
    float* fifo_zero_ref;
} Aec;

int  aec_create(Aec* a, const AecConfig* cfg);
void aec_destroy(Aec* a);
void aec_reset(Aec* a);

/**
 * Realign an AEC_DELAY_EXTERNAL_ALIGNED instance across a caller-side change
 * of the far alignment, WITHOUT the full reset that exposes echo again and
 * restarts the WOLA sequence (the spectrogram "vertical line").
 *
 * delta_samples = new_alignment - old_alignment: positive when the aligned
 * far stream advances (delay grew, including the first acquisition from raw
 * far where old_alignment is 0), negative when it retards.
 *
 * When the filter is demonstrably cancelling (aec_linear_is_cancelling(), the
 * same two-reading test the backward-delay quarantine uses -- NOT the
 * inst-ERLE ring alone, which ages out across a stretch of far silence and
 * would send a perfectly good filter down the reset path) and the shift fits
 * the tap span, the learned IR is shifted by delta and cancellation survives
 * the realign.
 *
 * Otherwise the filter is RESET -- taps, far spectra, far analysis history and
 * the convergence/mu/stationarity latches -- because taps that were not
 * shifted sit at the alignment the caller just abandoned: they do not "re-adapt
 * from a head start", they subtract a decorrelated copy of the echo (measured
 * 1.3-1.5x the echo, i.e. worse than no filter at all, held for hundreds of
 * hops by the converged-mu floor and the stationary-render freeze). The reset
 * is FILTER-ONLY: the synthesis overlap-add and the mic-side analysis frame
 * are untouched, so the output stays continuous -- that is the whole
 * difference from aec_reset(), whose extra reach is exactly the spectrogram
 * "vertical line". Reconvergence tracks an aec_reset() twin's trajectory.
 *
 * Either outcome bumps the alignment generation an AecLinearContext consumer
 * polls: the far this filter is matched to moved, so cross-hop far caches are
 * stale whichever branch ran.
 *
 * Direction asymmetry. A RETARD (delta < 0) also clears the filter's
 * per-partition far spectra -- the held history is ahead of the new stream and
 * convolving it against the shifted taps mis-estimates the echo -- so the echo
 * is exposed until the partition ring refills past the shifted response. The
 * bound is ceil((|delta| + the response's own tap offset) / hop_size) + 1
 * hops, i.e. at worst n_partitions + 1 when the response sits at the end of
 * the tap span; measured on a 256-sample hop with a 1024-sample span:
 * delta=-300 at tap 0 -> 2 hops, delta=-768 at tap 100 -> 4 hops, delta=-900
 * at tap 60 -> 4 hops. Within that window the residual is NOT bounded by the
 * echo either: partially-refilled history convolved against the shifted taps
 * measured 1.30x the echo on the -900 case. Cancellation resumes outright
 * afterwards. An ADVANCE (delta > 0) keeps the history and settles within a
 * hop; the far samples the caller's own jump skipped are unrecoverable by
 * either direction, since they were never presented to the filter.
 *
 * Returns 1 when the warm tap-transfer ran, 0 when the filter reset ran, and
 * -1 on a NULL instance or a mode other than AEC_DELAY_EXTERNAL_ALIGNED.
 * delta_samples == 0 is a no-op returning 0.
 */
int aec_apply_external_realign(Aec* a, int delta_samples);

/* Static-memory companions: place Aec + all backing arrays in a single pool.
 * aec_get_mem_size() returns the total byte requirement; caller allocates once.
 * aec_init() places the Aec struct at mem[0] and returns it; no malloc called. */
size_t aec_get_mem_size(const AecConfig* cfg);
Aec*   aec_init(void* mem, size_t mem_size, const AecConfig* cfg);

/* Process exactly hop_size samples — OFFLINE / lockstep path. Byte-exact to
 * Python aec.py. Render and capture supplied together. */
void aec_process(Aec* a, const float* mic, const float* ref, float* out);

/* Context-only variant of aec_process(): runs the full linear filter + AEC3
 * post/RES block (so aec_get_res_context() is fully populated afterward)
 * but never writes an `out` buffer -- for callers that only ever read the
 * context (error_spec / res_gain / formed_output / etc), never
 * aec_process()'s own returned audio. Cheaper per-hop than aec_process() by
 * exactly one O(hop) memcpy; the linear filter and RES/post cost is
 * identical either way.
 *
 * No mixing restriction: this and aec_process() / aec_process_capture() /
 * aec_process_context_shared_far() all advance identical state and differ
 * only by whether the result is copied out, so they may be interleaved
 * freely on one instance without an intervening aec_reset(). */
void aec_process_context(Aec* a, const float* mic, const float* ref);

/* Shared-far-end-FFT variant of aec_process_context(): shared_far_spec
 * (non-NULL, length n_freqs) lets this instance skip its own far-end FFT
 * and borrow one an external caller already computed -- for multi-instance
 * callers (e.g. a 4-lane wrapper) whose instances all see the identical
 * far-end signal every hop. shared_far_spec must be exactly the value the
 * computing instance's aec_get_res_context() would return this same hop,
 * computed from the SAME far-end time-domain signal this call's own `ref`
 * carries -- `ref` is still required even when shared_far_spec is supplied
 * (the OLA far-buffer history and every non-FFT use of the raw far signal
 * are unaffected by this sharing). Pass NULL to fall back to
 * aec_process_context()'s own behavior (compute this instance's own FFT)
 * -- see aec.c's doc comment on this function for the full precondition. */
void aec_process_context_shared_far(
        Aec* a, const float* mic, const float* ref,
        const Complex* shared_far_spec);

/* Shared-far-end-FFT instrumentation, read-only: how many hops (cumulative
 * since construction or the last aec_reset()) this instance has actually
 * run its own far-end FFT, as opposed to borrowing one -- either via the
 * internal shadow->main dedup or an external
 * aec_process_context_shared_far() caller. A lane driven exclusively
 * through aec_process_context_shared_far() with a valid spectrum every
 * hop stays at 0 forever; a lane computing its own (aec_process() /
 * aec_process_context()) increments by 1 every hop. Intended for tests/
 * instrumentation proving a multi-lane caller's total far-FFT count
 * actually dropped after switching lanes over to share one spectrum --
 * not part of the audio path, has no effect on any processing. */
long aec_far_fft_real_compute_count(const Aec* a);

/* Read-only: is this instance's linear filter demonstrably cancelling right
 * now? 1 = yes.
 *
 * The single definition of "cancelling" behind the backward-jump quarantine
 * (delay_backward_quarantine_enabled), exported so that a wrapper owning its
 * own shared delay estimator (pipelines/4ch_aec_bf_nr_res) asks the SAME question
 * of its lanes that this engine asks of itself, thresholds included, instead
 * of restating them. Reads two figures the engine already maintains -- last
 * hop's windowed ERLE against 2.5 dB, and the recent inst-ERLE peak against
 * cfg.delay_acquire_inst_erle_db -- so it adds nothing to the hot path.
 *
 * Both readings start at 0, so an instance whose ERLE machinery has not run
 * (or has just been reset) answers 0: "not demonstrably cancelling" is the
 * fail-open answer, never a blocking one. Correspondingly this is evidence
 * of cancellation, not its absence -- a 0 does NOT assert the filter is
 * broken, only that nothing here proves it is working. */
int aec_linear_is_cancelling(const Aec* a);

/* ── Streaming API (real-time async render/capture) ───────────────────────
 * For deployments where the far-end (render) and mic (capture) arrive on
 * separate calls / threads and not necessarily 1:1. aec_analyze_render()
 * buffers a render hop; aec_process_capture() consumes the delay-aligned
 * render and produces output. Both return the buffering event observed.
 *
 * THREADING CONTRACT (F09, SPSC only): exactly ONE thread may call
 * aec_analyze_render() and exactly ONE (possibly different) thread may call
 * aec_process_capture() on a given Aec — single-producer/single-consumer.
 * Under that contract the two functions may run fully concurrently with no
 * external lock; every field they share (fifo_write, fifo_read) is touched
 * only through acquire/release `__atomic_*_n` builtins (see the field
 * comments on the Aec struct above) and the ring's audio payload is always
 * memcpy'd by its producer before, and read by its consumer after, the
 * matching atomic hand-off — so there is no data race on the FIFO itself
 * for that one-producer/one-consumer shape.
 *
 * Anything outside that shape is NOT covered and needs the caller to add
 * its own external serialization:
 *   - a second render-side or second capture-side thread (two producers or
 *     two consumers) — fifo_write and the render/capture stats counters are
 *     single-writer-assumed and are not safe with a second writer on the
 *     same side;
 *   - aec_reset() / aec_destroy() while either thread might still call in —
 *     both do plain (non-atomic) field writes and assume the streaming
 *     pair is quiesced first;
 *   - mixing aec_process_capture() with the offline aec_process() on the
 *     same instance concurrently with the streaming pair.
 *
 * Lockstep equivalence: calling aec_analyze_render(ref) immediately followed
 * by aec_process_capture(mic, out) from a SINGLE thread (the degenerate,
 * always-supported case of the contract above) yields byte-identical output
 * to aec_process(mic, ref, out) — the FIFO is pass-through and no event
 * fires.
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
    /* Freq-domain seam for an external post-NR residual suppressor (mirrors the
     * Python AecResContext freq fields). Valid when enable_res=1 or
     * return_res_context=1; NULL otherwise. error_spec is the exact
     * 50%-overlap periodic-sqrt-Hann
     * STFT of the formed linear output E(f), in AUDIO scale. echo_spec is the
     * matching formed echo estimate and near_spec = error_spec + echo_spec is
     * the matching windowed capture spectrum. res_gain is the AEC3
     * SuppressionGain G_res(f), amplitude [0..1]. r2 = residual-echo PSD R²(f)
     * and comfort_noise = CNG N²(f), BOTH int16²-scaled (divide by 32768² to
     * reach the |E|² audio-power scale, as run_res does). All length n_freqs.
     * The pointers alias the AEC's internal per-hop buffers — read before
     * the next AEC processing call on this instance, whichever entry point
     * it is (aec_process(), aec_process_context(),
     * aec_process_context_shared_far(), or aec_process_capture()): every
     * one of them re-runs aec_process_core() and overwrites these buffers
     * in place, not just aec_process() specifically. */
    const Complex* error_spec;     /* reconstructing WOLA linear E(f), audio scale */
    const float*   res_gain;       /* AEC3 G_res(f), amplitude [0..1]; NULL when
                                     * AecConfig.spatial_linear_context is set
                                     * (this lane's G_res is never computed --
                                     * NULL signals "not computed", since the
                                     * underlying buffer is otherwise never
                                     * written past its zero-init and would
                                     * otherwise misread as "fully suppressed")*/
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
 *     UNLOCKED means the content is the RAW far (pre-acquisition, ring not
 *     yet filled, or delay estimation disabled) -- a consumer must not feed
 *     it to a small-search-range aligner as if it were aligned.
 *   - generation increments on every ring-offset change (first acquisition,
 *     confirmed shift including the flagless soft-recovery paths, reset) and
 *     saturates instead of wrapping. Poll it to invalidate cross-hop caches
 *     (far feature rings, attention histories); differencing delay_samples
 *     alone misses transient A->B->A shifts between polls.
 *   - CHANGED is reported exactly on the hop whose processing bumped
 *     generation; the next hop reads LOCKED again.
 *   - Out-of-range bulk delay is NOT detectable here. With a single dominant
 *     out-of-range path the estimator may remain UNLOCKED, but an in-range
 *     early reflection can also produce a confident lock at the wrong delay.
 *     The seam has no ground truth with which to distinguish that mis-lock;
 *     products must choose n from a measured route-delay envelope and own the
 *     fail-open/recovery policy. The getter itself mutates no state.
 *
 * PER-MODE SEMANTICS (cfg.delay_mode; productization plan appendix 11.1) --
 * the three modes make the previously implicit alignment state explicit,
 * and none of them changes a single output sample:
 *
 *   AEC_DELAY_MATCHED
 *     Unchanged: delay_samples is the applied ring offset (-1 before the
 *     first acquisition), delay_confidence is the estimator's own
 *     0.0/0.5/1.0, and generation bumps at every ring-offset change.
 *
 *   AEC_DELAY_FIXED
 *     delay_samples == cfg.fixed_delay_samples, always (never -1).
 *     delay_confidence == 1.0 -- the delay was MEASURED out of band, so it
 *     carries no estimator uncertainty. generation NEVER moves while
 *     processing (there is no estimator to re-lock); the only bump is
 *     aec_reset(), which genuinely invalidates the ring content a consumer
 *     may have cached. delay_state is LOCKED from the hop the ring first
 *     holds fixed_delay_samples + hop_size samples onward -- i.e. it is
 *     UNLOCKED for the first ceil(fixed_delay_samples / hop_size) hops. On
 *     the following process call, writing the current hop makes the ring
 *     hold fixed_delay_samples + hop_size samples, so that call already
 *     reports LOCKED and serves the requested offset. (Plan appendix 11.1 wrote
 *     "LOCKED" without qualifying that fill window; reporting LOCKED over
 *     raw far would break this seam's core promise -- "UNLOCKED means the
 *     content is the RAW far" -- for the ~fixed_delay_samples of audio a
 *     small-search-range consumer is most likely to mis-handle. Steady
 *     state is LOCKED as specified.)
 *
 *   AEC_DELAY_EXTERNAL_ALIGNED
 *     aligned_far_hop IS the caller's own far hop (no ring exists; the only
 *     transform it can have seen is the optional reference soft-clip).
 *     delay_samples == 0 and delay_state == LOCKED, because the caller has
 *     CONTRACTED that far is already aligned -- this is the one mode where
 *     "aligned" is an input, not a measurement. delay_confidence == 1.0 and
 *     generation never moves while PROCESSING -- there is no estimator to
 *     re-lock -- but aec_apply_external_realign() bumps it out of band, on
 *     both its warm and reset outcomes: that call is the caller telling this
 *     instance its contracted alignment just changed, which is exactly the
 *     event a cached far ring has to be invalidated on. (Before delay_mode existed
 *     this configuration -- enable_delay_est=0 -- reported delay_samples=-1
 *     / UNLOCKED, because the library could not tell "caller pre-aligned"
 *     apart from "estimator switched off and nobody aligned anything".
 *     The explicit mode is what makes the stronger statement legitimate.) */
typedef enum AecLinearDelayState {
    AEC_LINEAR_DELAY_UNLOCKED = 0,  /* far_hop is raw / not compensated */
    AEC_LINEAR_DELAY_LOCKED   = 1,  /* compensated with a stable offset */
    AEC_LINEAR_DELAY_CHANGED  = 2   /* compensated, but the offset changed
                                     * during THIS hop (flush far caches) */
} AecLinearDelayState;

typedef struct AecLinearContext {
    int hop_size;
    const float* formed_linear_hop;  /* formed linear error e_form when the
                                      * AEC3 post chain runs this config
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
 * Read-only status query for an integrator to poll (e.g. once per second)
 * to tell whether the pipeline is behaving on an embedded target. Fills a
 * plain struct from fields the engine already maintains — no logging inside
 * the library, no extra per-frame state, no added hot-path cost; the caller
 * (e.g. example/aec_wav.c --debug) is responsible for printing. */
typedef struct AecDebugStatus {
    /* delay estimation (EchoPathDelayEstimator / AEC3 matched-filter) */
    int    delay_samples;      /* current applied delay, samples; -1 = not yet
                                 * acquired (mirrors a->current_delay)         */
    float  delay_confidence;   /* MATCHED: the estimator's own 0.0/0.5/1.0
                                 * (delay_aec3_confidence). FIXED and
                                 * EXTERNAL_ALIGNED: 1.0 -- the alignment is
                                 * a caller contract (a bring-up measurement
                                 * / pre-aligned ref), not something this
                                 * library estimated, so there is no
                                 * uncertainty to report. Same value
                                 * AecLinearContext.delay_confidence gives.  */
    int    delay_updates;      /* matched-filter update count
                                 * (delay_aec3_n_updates); 0 outside MATCHED */

    /* linear filter health */
    float  erle_windowed_db;   /* windowed ERLE (dB) from the last hop that
                                 * ran the AEC3 post chain (a->last_erle_windowed),
                                 * 0.0 before the post chain has run once      */
    int    usable_linear;      /* AEC3 usable-linear-estimate gate state
                                 * (aec_state_usable_linear_estimate)          */
    int    filter_converged;   /* FilterConvergenceAnalyzer latch state
                                 * (a->convergence.converged, same value
                                 * AecResContext.filter_converged exposes)     */

    /* levels — EMA (alpha=0.95), NOT an instantaneous last-hop mean. Reuses
     * the power-tracking EMAs the ERLE machinery already updates every hop
     * (two multiply-accumulate loops that run unconditionally), so reading
     * them here adds zero per-frame cost. */
    float  near_power;         /* EMA of mic/near-hop power (a->near_power)    */
    float  out_power;          /* EMA of the linear, pre-RES-gain output power
                                 * (a->raw_error_power)                        */

    /* duty-cycle engagement census, cumulative since aec_create/aec_init/
     * aec_reset. duty_hops_run / duty_hops_total is the MEASURED
     * matched-filter engagement (see the duty_hops_* doc on the Aec struct
     * above). Both are 0 whenever this instance never constructs a matched
     * filter to decimate -- delay_mode != AEC_DELAY_MATCHED, not only the
     * legacy enable_delay_est=0 case -- since there is then no cost for the
     * duty machine to have saved. */
    unsigned long long duty_hops_total;
    unsigned long long duty_hops_run;
} AecDebugStatus;

void aec_debug_status(const Aec* a, AecDebugStatus* out);

/* ── Static-pool memory breakdown (plan §3.4.6 RAM acceptance test 6) ─────
 * `aec_get_mem_size()` returns one total; this labels two subsets of that
 * same total instead of a caller (e.g. a CLI's --print-mem-size) having to
 * re-derive them from the config by hand -- which would be exactly the
 * kind of second, driftable size computation the pool-first refactor (plan
 * §3) exists to avoid. total_bytes is always aec_get_mem_size(cfg) itself,
 * called from inside this function, so the two can never disagree.
 *
 *   estimator_bytes  delay_aec3_get_mem_size() for the resolved
 *                    (sample_rate, hop_size, delay_num_filters) triple;
 *                    0 outside AEC_DELAY_MATCHED (no estimator is built).
 *   ring_bytes       the alignment-ring float buffer, mode-aware
 *                    (see aec_ref_ring_samples() in aec.c / the manual's
 *                    ring-size-formula table); 0 for
 *                    AEC_DELAY_EXTERNAL_ALIGNED (no ring at all).
 *
 * Returns 0 (and zeroes *out) under exactly the same condition
 * aec_get_mem_size() itself reports failure on: cfg is NULL, or
 * aec_validate_config(cfg) rejects it. */
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
