/* initial_state.c — C port of python/modules/state/initial_state.py.
 *
 * Cold-start state machine (initial-phase counter). No float arithmetic in the
 * update path; the sole numeric op is the constructor threshold
 *   int(initial_state_seconds * HOPS_PER_SECOND)
 * = a float32 multiply (float32-by-design; converted for uniformity, drift
 * accepted) then truncate-toward-zero (Python int()), reproduced as
 * (int)((float)seconds * HOPS_PER_SECOND). All run-time state is integer/bool.
 */
#include "initial_state.h"
#include "aec3_scale.h"

void initial_state_init(InitialState *s, int conservative_initial_phase,
                        float initial_state_seconds,
                        int hop_size, int sample_rate) {
    s->conservative = conservative_initial_phase ? 1 : 0;
    /* Was `(int)(initial_state_seconds * INITIAL_STATE_HOPS_PER_SECOND)` with
     * HOPS_PER_SECOND frozen at 100 (hop=160/sr=16000); computed live via
     * aec3_ms_to_hops so the window stays initial_state_seconds at any grid. */
    s->initial_state_hops = aec3_ms_to_hops(
        initial_state_seconds * 1000.0f, hop_size, sample_rate);
    /* AEC3 5 s conservative threshold. */
    s->conservative_hops = aec3_ms_to_hops(5000.0f, hop_size, sample_rate);
    s->initial_state = 1;
    s->transition_triggered = 0;
    s->strong_not_saturated_render_blocks = 0;
}

void initial_state_reset(InitialState *s) {
    s->initial_state = 1;
    s->transition_triggered = 0;
    s->strong_not_saturated_render_blocks = 0;
}

void initial_state_update(InitialState *s, int active_render, int saturated_capture) {
    int prev_initial;
    if (active_render && !saturated_capture) {
        /* Threshold-gate counter: the only two readers below each test
         * "< threshold" against exactly one of conservative_hops (fixed
         * per-instance since s->conservative never changes post-construction)
         * or initial_state_hops (config-derived, not a hardcoded compile-time
         * constant). Saturate at the larger of the two -- always >= the one
         * branch actually evaluated for this instance's lifetime -- so the
         * comparison result matches the unbounded counter identically
         * forever, without ever risking signed-integer-overflow UB on a very
         * long streaming session. */
        int cap = (s->conservative_hops > s->initial_state_hops)
                      ? s->conservative_hops : s->initial_state_hops;
        if (s->strong_not_saturated_render_blocks < cap) {
            s->strong_not_saturated_render_blocks += 1;
        }
    }
    prev_initial = s->initial_state;
    if (s->conservative) {
        s->initial_state =
            (s->strong_not_saturated_render_blocks < s->conservative_hops) ? 1 : 0;
    } else {
        s->initial_state =
            (s->strong_not_saturated_render_blocks < s->initial_state_hops) ? 1 : 0;
    }
    s->transition_triggered = (!s->initial_state) && prev_initial ? 1 : 0;
}

int initial_state_initial_state_active(const InitialState *s) {
    return s->initial_state;
}

int initial_state_transition_triggered(const InitialState *s) {
    return s->transition_triggered;
}
