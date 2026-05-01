/**
 * render_activity.h - Far-end activity + stationarity detector
 * Port of Python aec.py RenderActivityDetector (lines 2299-2358).
 */

#ifndef RENDER_ACTIVITY_H
#define RENDER_ACTIVITY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RenderActivityDetector RenderActivityDetector;

typedef struct {
    float far_pwr;          /* mean(far²) + 1e-10 */
    int   is_active;        /* sticky: True after first audible frame */
    int   is_stationary;    /* CV² < 0.02 */
    int   warmup_active;    /* mean(far²) > 1e-6 raw */
} RenderActivityState;

RenderActivityDetector* ra_create(void);
void ra_destroy(RenderActivityDetector* ra);
void ra_reset(RenderActivityDetector* ra);

/**
 * Update from far_end[hop]. Returns RenderActivityState by value.
 */
RenderActivityState ra_update(RenderActivityDetector* ra,
                              const float* far_end, int hop);

int ra_is_active(const RenderActivityDetector* ra);
int ra_is_stationary(const RenderActivityDetector* ra);

#ifdef __cplusplus
}
#endif
#endif /* RENDER_ACTIVITY_H */
