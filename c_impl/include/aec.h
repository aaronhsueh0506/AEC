/* aec.h — top-level v3.22 AEC orchestration (AEC3 post-filter chain).
 *
 * Wires the bit-exact v3.22 modules into one frame-level aec_process() that
 * mirrors python/modules/orchestrator.py AEC.process for the BALANCED preset
 * (mode=PBFDKF, simple variable-mu, enable_res=True,
 * enable_cng per preset). The post-stage is driven entirely by aec3_post_run
 * (the bit-exact full _aec3_post orchestration); this file ports the AUDIO
 * PATH of process() around it.
 *
 * Out of scope for this cutover:
 *   - AecMode != PBFDKF (NLMS / buffered FDAF queue)
 *   - get_stats / debug logger / capture_stages / pure-diagnostic _diag stashes
 *   - (historical note: aec_init/aec_get_mem_size static-memory pool is now
 *     IMPLEMENTED — see STATIC_MEMORY.md; both paths live in aec.c.)
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

typedef struct AecConfig {
    int    sample_rate;        /* 16000 */
    int    fft_size;           /* frame size; 256/512 @16k, 1024 @48k */
    int    filter_length;      /* 832 samples (52 ms) */
    int    n_partitions;       /* derived if 0 */
    float  mu;                 /* 0.3 */
    float  delta;              /* 1e-8 */

    /* toggles */
    int    enable_cng;             /* 1 */
    int    enable_delay_est;       /* 1 */
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
     * construction time, the one place guaranteed to see the FINAL
     * resolved hop after any caller override (e.g. aec_wav.c's
     * --fft-size), NOT aec_config_defaults() (2026-08 gap-fix) -- the
     * values in comments below are the legacy-grid figures, not
     * necessarily today's default. */
    float  shadow_err_alpha;       /* 0.80 @ 10 ms hop */
    float  shadow_mu_min;          /* 0.5  */
    float  shadow_mu_nlms;         /* 0.5  */
    int    warmup_frames;          /* 100 hops = 1000 ms @ 10 ms hop */
    int    epc_hangover;           /* 20 hops = 200 ms @ 10 ms hop */
    float  epc_total_rise;         /* 1.5  */
    float  epc_delta_threshold;    /* 0.3  */
    float  epc_mu_floor;           /* 0.5  */
    float  max_delay_ms;           /* 1024 */
    float  delay_buffer_ms;        /* 2048 */
    float  delay_est_init_s;       /* 0.3  */
    float  delay_est_period_s;     /* 0.5  */
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
     * authored @ 10 ms hop and retimed live in aec_carve() (2026-08
     * gap-fix); ne_recent_sustain is a genuine consecutive-event count
     * (NOT a duration) and is intentionally left unretimed. */
    float  ne_recent_threshold;               /* 0.3 */
    int    ne_recent_hold;                    /* 150 hops = 1500 ms @ 10 ms hop */
    int    ne_recent_sustain;                 /* 3 (event count, not retimed) */

    /* preset strength axis (the ONLY field mild/balanced/aggressive differ
     * in): SuppressionGain far-active min-gain floor in dB
     * (mild -20 / balanced -28 / aggressive -38). */
    float  min_gain_floor_far_active_db;

    /* FilterMisadjustmentEstimator (AEC3 ScaleFilter). stable/hangover_frames
     * are wall-clock-authored @ 10 ms hop and retimed live in aec_carve()
     * (2026-08 gap-fix). */
    int    filter_misadjustment_stable_frames;   /* 30 hops = 300 ms @ 10 ms hop */
    int    filter_misadjustment_hangover_frames;  /* 100 hops = 1000 ms @ 10 ms hop */
    float  filter_misadjustment_scale_min;        /* 0.5 */
    float  filter_misadjustment_scale_max;        /* 2.0 */
} AecConfig;

void aec_config_defaults(AecConfig* cfg, int sample_rate);
void aec_config_from_preset(AecConfig* cfg, AecPreset preset, int sample_rate);

/* Sample-rate whitelist query (F05: no sample-rate validation anywhere).
 * The single source of truth for the whitelist lives in aec.c next to
 * aec_validate_config(); this is the public read of it, for callers (e.g.
 * example/aec_wav.c) that need to reject an unsupported input WAV rate
 * before ever constructing an AecConfig. Returns 1 iff sample_rate is
 * production-qualified today (8000/16000/48000 — see aec.c for the
 * per-rate table this rests on). */
int aec_is_valid_sample_rate(int sample_rate);

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

    /* delay (AEC3 matched filter) + ring */
    DelayAec3 delay;       int has_delay;
    float* ref_ring;       int ref_ring_size, ref_ring_write, ref_ring_filled;
    int    current_delay;  /* -1 until first acquisition */
    int    pending_delay;  int has_pending; int pending_delay_ttl;
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
    float  limiter_gain;
    float* limiter_near_lag;   int has_limiter_lag;
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
    /* F09 Variant A' (drop-new + consumer catch-up): a from-scratch SPSC
     * ring, replacing an earlier design where both the producer and the
     * consumer could advance the read cursor (the producer's overrun path
     * used to drop the OLDEST buffered hop, which requires a second writer
     * on the read side and forces every touch of it through a full
     * fetch_add RMW). This rewrite gives each cursor exactly ONE writer —
     * the textbook SPSC shape — by moving the overrun response to "producer
     * drops the NEW incoming hop" (drop-new) and adding a symmetric
     * "consumer catches up to the freshest hop" response for when the
     * consumer has fallen a full `cap` hops behind. The same two
     * AecBufferingEvent values are still reported; only which side's data is
     * sacrificed under sustained skew changes (newest vs. oldest hop
     * dropped). aec.c touches every cross-thread field below through
     * GCC/Clang `__atomic_*_n` builtins on plain int/unsigned lvalues, never
     * `<stdatomic.h>` _Atomic types — so the struct's size and member
     * offsets are completely unaffected (no ABI break for existing callers/
     * pool-size math) and the header stays includable from a C89
     * translation unit.
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

    /* F03 arena-fication: aec_create()'s single malloc'd arena backing every
     * sub-module array above (ref_ring, render_fifo, rsa_counters, the main/
     * shadow filter state, the whole AEC3 post chain, hop/per-hop-freq-bin
     * scratch, ...). NULL on the static-pool path (aec_init) — there the
     * caller owns the pool and aec_destroy() must not free it. Appended at
     * the end of the struct so existing field offsets are unchanged. */
    void*  heap_arena;

    /* Appended AFTER heap_arena (same append-at-end precedent, so every
     * field above keeps its pre-existing offset). fifo_zero_ref: an
     * immutable all-zero hop, the F09 Variant A' underrun reference — see
     * the streaming-FIFO design note above. */
    float* fifo_zero_ref;
} Aec;

int  aec_create(Aec* a, const AecConfig* cfg);
void aec_destroy(Aec* a);
void aec_reset(Aec* a);

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
 * but skips the output limiter and never writes an `out` buffer -- for
 * callers that only ever read the context (error_spec / res_gain /
 * formed_output / etc), never aec_process()'s own returned audio. Cheaper
 * per-hop than aec_process() by exactly the limiter's O(hop) work plus one
 * O(hop) memcpy; the linear filter and RES/post cost is identical either
 * way. See aec.c's aec_process_context() doc comment for the one hard
 * precondition: a single Aec instance must use ONLY aec_process() or ONLY
 * aec_process_context() for its whole lifetime, never both interleaved
 * (mixing desyncs the limiter's one-hop state, producing an audible gain
 * discontinuity the next aec_process() call). */
void aec_process_context(Aec* a, const float* mic, const float* ref);

/* Group 6: like aec_process_context(), but shared_far_spec (non-NULL,
 * length n_freqs) lets this instance skip its own far-end FFT and borrow
 * one an external caller already computed -- for multi-instance callers
 * (e.g. a 4-lane wrapper) whose instances all see the identical far-end
 * signal every hop. shared_far_spec must be exactly the value the
 * computing instance's aec_get_res_context() would return this same hop,
 * computed from the SAME far-end time-domain signal this call's own `ref`
 * carries -- `ref` is still required even when shared_far_spec is
 * supplied (the OLA far-buffer history and every non-FFT use of the raw
 * far signal are unaffected by Group 6). Pass NULL to fall back to
 * aec_process_context()'s own behavior (compute this instance's own
 * FFT) -- see aec.c's doc comment on this function for the full
 * precondition and aec_process_context()'s doc comment for the shared
 * one-entry-point-per-instance-lifetime rule. */
void aec_process_context_shared_far(
        Aec* a, const float* mic, const float* ref,
        const Complex* shared_far_spec);

/* Group 6 instrumentation, read-only: how many hops (cumulative since
 * construction or the last aec_reset()) this instance has actually run
 * its own far-end FFT, as opposed to borrowing one -- either via the
 * internal shadow->main dedup or an external
 * aec_process_context_shared_far() caller. A lane driven exclusively
 * through aec_process_context_shared_far() with a valid spectrum every
 * hop stays at 0 forever; a lane computing its own (aec_process() /
 * aec_process_context()) increments by 1 every hop. Intended for tests/
 * instrumentation proving a multi-lane caller's total far-FFT count
 * actually dropped after switching lanes over to share one spectrum --
 * not part of the audio path, has no effect on any processing. */
long aec_far_fft_real_compute_count(const Aec* a);

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
     * The pointers alias the AEC's internal per-hop buffers — read before the
     * next aec_process() call. */
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
    float  delay_confidence;   /* 0.0 / 0.5 / 1.0 (delay_aec3_confidence)      */
    int    delay_updates;      /* matched-filter update count
                                 * (delay_aec3_n_updates)                      */

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
} AecDebugStatus;

void aec_debug_status(const Aec* a, AecDebugStatus* out);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_H */
