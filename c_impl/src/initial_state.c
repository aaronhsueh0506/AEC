/* initial_state.c -- see initial_state.h. No float arithmetic in the update path. */
#include "initial_state.h"
#include "aec_scale.h"

void initial_state_init(InitialState *s, int conservative_initial_phase,
                        float initial_state_seconds,
                        int hop_size, int sample_rate) {
    s->conservative = conservative_initial_phase ? 1 : 0;
    /* Thresholds are derived from the live grid, so the window is initial_state_seconds at any
     * rate. */
    s->initial_state_hops = aec_ms_to_hops(
        initial_state_seconds * 1000.0f, hop_size, sample_rate);
    /* 5 s conservative threshold. */
    s->conservative_hops = aec_ms_to_hops(5000.0f, hop_size, sample_rate);
    s->cap_hops = (s->conservative_hops > s->initial_state_hops)
                      ? s->conservative_hops : s->initial_state_hops;
    s->active_threshold_hops =
        s->conservative ? s->conservative_hops : s->initial_state_hops;
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
        /* The counter saturates at the larger threshold so it cannot overflow. */
        if (s->strong_not_saturated_render_blocks < s->cap_hops) {
            s->strong_not_saturated_render_blocks += 1;
        }
    }
    prev_initial = s->initial_state;
    s->initial_state =
        (s->strong_not_saturated_render_blocks < s->active_threshold_hops) ? 1 : 0;
    s->transition_triggered = (!s->initial_state) && prev_initial ? 1 : 0;
}

int initial_state_transition_triggered(const InitialState *s) {
    return s->transition_triggered;
}
