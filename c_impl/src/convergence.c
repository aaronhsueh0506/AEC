/**
 * convergence.c - port of Python FilterConvergenceAnalyzer (aec.py 2369-2434).
 */

#include "convergence.h"
#include <stdlib.h>
#include <math.h>

#define CONV_ERLE_DB  5.0f
#define CONV_FRAMES   10
#define DIV_ERLE_LIN  0.63f
#define DIV_ALPHA     0.9f
#define DIV_DECAY     0.95f


struct FilterConvergenceAnalyzer {
    int   converged;
    int   once_converged;
    int   conv_counter;
    float divergence;
};


FilterConvergenceAnalyzer* fc_create(void) {
    FilterConvergenceAnalyzer* fc = (FilterConvergenceAnalyzer*)calloc(1, sizeof(FilterConvergenceAnalyzer));
    if (!fc) return NULL;
    fc_reset(fc);
    return fc;
}


void fc_destroy(FilterConvergenceAnalyzer* fc) { free(fc); }


void fc_reset(FilterConvergenceAnalyzer* fc) {
    if (!fc) return;
    fc->converged = 0;
    fc->once_converged = 0;
    fc->conv_counter = 0;
    fc->divergence = 0.0f;
}


void fc_mark_diverged(FilterConvergenceAnalyzer* fc) {
    if (!fc) return;
    fc->converged = 0;
    fc->conv_counter = 0;
}


void fc_update_divergence(FilterConvergenceAnalyzer* fc,
                          float near_power, float raw_error_power) {
    if (!fc) return;
    if (fc->converged && near_power > 1e-8f) {
        float inst = near_power / (raw_error_power + 1e-10f);
        float is_div = (inst < DIV_ERLE_LIN) ? 1.0f : 0.0f;
        fc->divergence = DIV_ALPHA * fc->divergence + (1.0f - DIV_ALPHA) * is_div;
    } else {
        fc->divergence *= DIV_DECAY;
    }
}


int fc_update_convergence(FilterConvergenceAnalyzer* fc,
                          float near_power, float raw_error_power,
                          int far_active, int warmup_done) {
    if (!fc) return 0;
    if (fc->converged || near_power <= 1e-8f || !warmup_done || !far_active) {
        return 0;
    }
    /* Python: 10 * np.log10(near / err+eps) — use float32 log10 (log10f). */
    float ratio = near_power / (raw_error_power + 1e-10f);
    float inst_erle_db = 10.0f * log10f(ratio);
    if (inst_erle_db > CONV_ERLE_DB) {
        fc->conv_counter++;
    } else {
        fc->conv_counter = 0;
    }
    if (fc->conv_counter >= CONV_FRAMES) {
        fc->converged = 1;
        fc->once_converged = 1;
        return 1;
    }
    return 0;
}


int   fc_is_converged(const FilterConvergenceAnalyzer* fc) { return fc ? fc->converged : 0; }
int   fc_is_once_converged(const FilterConvergenceAnalyzer* fc) { return fc ? fc->once_converged : 0; }
float fc_get_divergence(const FilterConvergenceAnalyzer* fc) { return fc ? fc->divergence : 0.0f; }
