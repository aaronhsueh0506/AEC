/* subband_erle.h — C port of python/modules/state/subband_erle.py.
 *
 * SubbandErleEstimator: per-bin ERLE (echo-return-loss-enhancement) estimator
 * with onset compensation. Single-channel port (AEC3 SubbandErleEstimator).
 *
 * update(x2,y2,e2,converged_filter,coh_gate_mask):
 *   - gated accumulation of y2/e2 over kPointsToAccumulate(=6) points (only
 *     when converged_filter); low_render flag |= x2 < x2_band_threshold;
 *   - every 6th point: new_erle = y2_acc / max(e2_acc,1e-30) per interior bin,
 *     gated by e2_acc>0 (and optional E2/Y2 gate + coherence gate mask);
 *   - onset detection: coming_onset → hold window; per-bin smoothed EMA
 *     (alpha_up=0.05 rising / alpha_down=0.1 or 0 if low-render falling);
 *   - three EMAs maintained: _erle (cap min..max_erle[k]),
 *     _erle_onset_compensated (same caps), _erle_unbounded (cap min..1e5);
 *   - per-hop onset release decay of _erle_onset_compensated.
 *   - endpoints (bin 0, n-1) mirror their interior neighbour.
 *
 * PARITY (numpy 1.26 -> C):
 *   Real pipeline dtypes (captured on a real doubletalk case):
 *     x2 / y2 / e2          : float32 (257,)
 *     converged_filter      : bool scalar
 *     coh_gate_mask         : bool (257,)  (present, not None)
 *     _erle / _erle_onset_compensated / _erle_unbounded /
 *     _erle_during_onsets / _max_erle / _y2_acc / _e2_acc : float32 (257,)
 *     _coming_onset / _low_render_energy                  : bool (257,)
 *     _hold_counters                                      : int32 (257,)
 *     _x2_band_energy_threshold                           : float64 scalar
 *   All per-bin ERLE arithmetic is float32: numpy 1.x casts the Python-float
 *   scalars (alpha, min_erle, max_erle[k], 1e-30, 10.0) to float32 before
 *   combining with the float32 arrays, so every smoothing step rounds in
 *   float. The single float64 quantity is the x2 band-energy threshold; the
 *   comparison `x2(f32) < thr(f64)` promotes x2 to f64 (numpy mixed compare).
 *   Built with -ffp-contract=off so the per-op float roundings are not fused.
 *
 * STORAGE: caller supplies all per-bin arrays (length n_bins each). Sizes:
 *   erle, erle_oc, erle_unb, erle_during, max_erle, y2_acc, e2_acc : float
 *   coming_onset, low_render                                       : unsigned char
 *   hold_counters                                                  : int32_t
 */
#ifndef SUBBAND_ERLE_H
#define SUBBAND_ERLE_H

#include <stdint.h>

/* ── Constants (16 kHz / hop=160 reference, mirrored from subband_erle.py) ── */
#define SE_HOPS_PER_SECOND        100
/* _BLOCKS_TO_HOLD_ERLE        = int(0.4 * 100) = 40 */
#define SE_BLOCKS_TO_HOLD_ERLE    40
/* _BLOCKS_FOR_ONSET_DETECTION = 40 + int(0.6 * 100) = 100 */
#define SE_BLOCKS_FOR_ONSET_DETECTION 100
#define SE_POINTS_TO_ACCUMULATE   6
#define SE_UNBOUNDED_ERLE_MAX     100000.0
/* AEC3 kX2BandEnergyThreshold source value (scaled per hop in init). */
#define SE_X2_BAND_ENERGY_THRESHOLD 44015068.0

typedef struct {
    int    n_bins;
    float  min_erle;          /* _min_erle (Python float -> stored f32-cast at use) */
    int    use_onset_detection;
    int    use_min_erle_during_onsets;
    double alpha_up;          /* 0.05 (Python float) */
    double alpha_down;        /* 0.1  (Python float) */
    double onset_release_decay; /* 0.97 (Python float) */
    int    e2y2_gate_enabled;
    float  e2y2_gate_threshold; /* 0.5 */
    double x2_band_energy_threshold; /* float64 scalar (per_bin_psd_threshold) */

    /* per-bin state (caller-owned, length n_bins) */
    float   *max_erle;        /* LF half = max_erle_l, HF half = max_erle_h */
    float   *erle;
    float   *erle_onset_compensated;
    float   *erle_unbounded;
    float   *erle_during_onsets;
    unsigned char *coming_onset;     /* bool */
    int32_t *hold_counters;
    float   *y2_acc;
    float   *e2_acc;
    unsigned char *low_render_energy; /* bool */
    int      num_points;
} SubbandErle;

/* Storage layout (caller provides; sizes in elements, all length n_bins):
 *   max_erle, erle, erle_onset_compensated, erle_unbounded,
 *   erle_during_onsets, y2_acc, e2_acc          : float
 *   coming_onset, low_render_energy             : unsigned char
 *   hold_counters                               : int32_t
 *
 * hop_size is used only to scale the x2 band-energy threshold
 * (per_bin_psd_threshold, ref_hop=160). Mirrors the Python __init__ defaults:
 *   min_erle=1.0, max_erle_l=4.0, max_erle_h=1.5,
 *   use_onset_detection=1, use_min_erle_during_onsets=1,
 *   e2y2_gate_enabled=0, e2y2_gate_threshold=0.5.
 */
void subband_erle_init(SubbandErle *s, int n_bins,
                       float min_erle, float max_erle_l, float max_erle_h,
                       int use_onset_detection, int use_min_erle_during_onsets,
                       int hop_size, int e2y2_gate_enabled,
                       float e2y2_gate_threshold,
                       float *max_erle_st, float *erle_st, float *erle_oc_st,
                       float *erle_unb_st, float *erle_during_st,
                       unsigned char *coming_onset_st, int32_t *hold_st,
                       float *y2_acc_st, float *e2_acc_st,
                       unsigned char *low_render_st);

void subband_erle_reset(SubbandErle *s);

/* update(x2,y2,e2,converged_filter,coh_gate_mask).
 * coh_gate_mask may be NULL (matches Python coh_gate_mask=None). */
void subband_erle_update(SubbandErle *s,
                         const float *x2, const float *y2, const float *e2,
                         int converged_filter,
                         const unsigned char *coh_gate_mask);

/* erle(onset_compensated): returns pointer to erle_onset_compensated when
 * onset_compensated && use_onset_detection, else the plain erle array. */
const float *subband_erle_erle(const SubbandErle *s, int onset_compensated);
const float *subband_erle_erle_unbounded(const SubbandErle *s);
const float *subband_erle_erle_during_onsets(const SubbandErle *s);

#endif /* SUBBAND_ERLE_H */
