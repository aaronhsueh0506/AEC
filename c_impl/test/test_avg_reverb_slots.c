/* Ring-slot selection of the average-render reverb: the delayed partition and
 * the one just older must be distinct from the newest partition for every ring
 * size, write position and delay, including delays at and beyond the ring
 * length. */
#include "aec_post.h"

#include <stdio.h>

static int fails;

#define CHECK(cond, ...) do { if (!(cond)) { ++fails; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* Reference walk: step back from the newest partition one slot at a time. */
static int step_back(int n_part, int from, int steps) {
    int i = from;
    while (steps-- > 0) i = (i + n_part - 1) % n_part;
    return i;
}

int main(void) {
    int n_part, pidx, delay;
    for (n_part = 2; n_part <= 9; ++n_part) {
        for (pidx = 0; pidx < 3 * n_part; ++pidx) {
            for (delay = -3; delay <= n_part + 3; ++delay) {
                int di = -1, pi = -1;
                int newest = (pidx + n_part - 1) % n_part;
                int d = delay < 0 ? 0 : (delay > n_part - 2 ? n_part - 2 : delay);
                aec_post_avg_reverb_slots(n_part, pidx, delay, &di, &pi);
                CHECK(di == step_back(n_part, newest, d),
                      "delay slot n=%d pidx=%d delay=%d got %d", n_part, pidx, delay, di);
                CHECK(pi == step_back(n_part, newest, d + 1),
                      "past slot n=%d pidx=%d delay=%d got %d", n_part, pidx, delay, pi);
                CHECK(pi != newest,
                      "past slot aliases the newest partition n=%d pidx=%d delay=%d", n_part, pidx, delay);
            }
        }
    }
    {
        int di = -1, pi = -1;
        aec_post_avg_reverb_slots(7, 1, 6, &di, &pi);   /* delay = n_part - 1 */
        CHECK(pi != 0, "n=7 newest=0 delay=6: past slot must not be the newest, got %d", pi);
    }
    {
        int di = -1, pi = -1;
        aec_post_avg_reverb_slots(1, 5, 3, &di, &pi);   /* single-slot ring */
        CHECK(di == 0 && pi == 0, "n=1 slots %d %d", di, pi);
    }
    if (fails) { printf("test_avg_reverb_slots: %d failure(s)\n", fails); return 1; }
    printf("test_avg_reverb_slots: PASS\n");
    return 0;
}
