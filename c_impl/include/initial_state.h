/* initial_state.h — C port of python/modules/state/initial_state.py
 * (which mirrors docs/aec3_extracts/src/aec3/aec_state.cc:327-353).
 *
 * AecState::InitialState — cold-start state machine. Tracks whether the filter
 * is still in the initial-state regime; transitions out once enough
 * "strong, not saturated" render frames have accumulated.
 *
 *   conservative=true  : 5 s  (5 * HOPS_PER_SECOND hops)
 *   conservative=false : initial_state_seconds * HOPS_PER_SECOND hops
 *
 * PARITY (numpy 1.26 → C):
 *   This module carries NO float arithmetic in its update path. The only
 *   numeric op is the constructor threshold
 *     _initial_state_hops = int(initial_state_seconds * HOPS_PER_SECOND)
 *   which is a float32 multiply (float32-by-design; converted for uniformity,
 *   drift accepted — formerly a Python f64 multiply) followed by int()
 *   truncation toward zero. Reproduced as (int)((float)seconds * HOPS_PER_SECOND).
 *   All run-time state is integer/bool:
 *     _strong_not_saturated_render_blocks : int (monotonic counter)
 *     _initial_state, _transition_triggered : bool.
 */
#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

/* 10 ms hop @ 16 kHz; helper for AEC3-second thresholds (modules/state/_constants.py) */
#define INITIAL_STATE_HOPS_PER_SECOND 100

typedef struct {
    int conservative;                       /* bool _conservative              */
    int initial_state_hops;                 /* int(initial_state_seconds*HPS)  */
    int conservative_hops;                  /* 5 * HOPS_PER_SECOND             */
    int initial_state;                      /* bool _initial_state             */
    int transition_triggered;               /* bool _transition_triggered      */
    int strong_not_saturated_render_blocks; /* int counter                     */
} InitialState;

/* conservative_initial_phase: 0/1 ; initial_state_seconds: config float (default 2.5) */
void initial_state_init(InitialState *s, int conservative_initial_phase,
                        float initial_state_seconds);

void initial_state_reset(InitialState *s);

/* update(active_render, saturated_capture) — both bool (0/1). */
void initial_state_update(InitialState *s, int active_render, int saturated_capture);

int initial_state_initial_state_active(const InitialState *s);
int initial_state_transition_triggered(const InitialState *s);

#endif /* INITIAL_STATE_H */
