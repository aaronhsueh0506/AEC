/* Cold-start state machine: leaves the initial state once enough strong, unsaturated render
 * hops have accumulated (conservative mode: 5 s, otherwise initial_state_seconds).
 * Run-time state is integer only. Python reference: modules/state/initial_state.py. */
#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

typedef struct {
    int conservative;
    int initial_state_hops;
    int conservative_hops;
    int cap_hops;
    int active_threshold_hops;
    int initial_state;
    int transition_triggered;
    int strong_not_saturated_render_blocks;
} InitialState;

/* conservative_initial_phase: 0/1. initial_state_seconds: default 2.5.
 * hop_size/sample_rate: live grid used to convert times to hops. */
void initial_state_init(InitialState *s, int conservative_initial_phase,
                        float initial_state_seconds,
                        int hop_size, int sample_rate);

void initial_state_reset(InitialState *s);

/* active_render and saturated_capture are 0/1. */
void initial_state_update(InitialState *s, int active_render, int saturated_capture);

int initial_state_transition_triggered(const InitialState *s);

#endif
