/* Force the allocator alignment that ILP32 malloc is allowed to provide. */
#include "aec.h"
#include "aec_white_box.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *alloc_base, *alloc_payload;
static size_t requested_alignment;
static int fail_alloc, allocations, releases;

static void *test_malloc(size_t size) {
    void *base = malloc(size + 32u);
    if (!base) return NULL;
    alloc_base = base;
    alloc_payload = (void *)((((uintptr_t)base + 15u) & ~(uintptr_t)15u) + 8u);
    ++allocations;
    return alloc_payload;
}

static int test_posix_memalign(void **out, size_t alignment, size_t size) {
    requested_alignment = alignment;
    if (fail_alloc) return ENOMEM;
    int rc = posix_memalign(out, alignment, size);
    if (!rc) {
        alloc_base = alloc_payload = *out;
        ++allocations;
    }
    return rc;
}

static void test_free(void *ptr) {
    if (ptr && ptr == alloc_payload) {
        free(alloc_base);
        alloc_base = alloc_payload = NULL;
        ++releases;
    } else {
        free(ptr);
    }
}

#define malloc test_malloc
#define posix_memalign test_posix_memalign
#define free test_free
#include "../src/aec.c"
#undef free
#undef posix_memalign
#undef malloc

int main(void) {
    int failures = 0;
    void *probe = test_malloc(32u);
    if (!probe || ((uintptr_t)probe & 15u) != 8u) return 2;
    test_free(probe);
    allocations = releases = 0;

    static const int grids[][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    for (size_t i = 0; i < sizeof(grids) / sizeof(grids[0]); ++i) {
        AecConfig cfg;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, grids[i][0]);
        cfg.fft_size = grids[i][1];
        Aec *a = NULL;
        const int allocations_before = allocations;
        requested_alignment = 0;
        fail_alloc = 1;
        if ((a = aec_create(&cfg)) != NULL || requested_alignment != 16u || alloc_payload ||
            allocations != allocations_before) {
            fprintf(stderr, "FAIL allocation rejection grid %zu\n", i);
            ++failures;
        }
        fail_alloc = 0;
        requested_alignment = 0;
        if ((a = aec_create(&cfg)) == NULL) {
            fprintf(stderr, "FAIL aligned create grid %zu\n", i);
            ++failures;
            test_free(alloc_payload);
            continue;
        }
        /* one block holds the instance: it starts the allocation, 16-byte aligned */
        if (requested_alignment != 16u || (void*)a != alloc_payload || a->heap_arena != (void*)a ||
            ((uintptr_t)a & 15u) != 0u || allocations != allocations_before + 1) {
            fprintf(stderr, "FAIL explicit alignment grid %zu\n", i);
            ++failures;
        }
        aec_destroy(a);
        if (alloc_payload || allocations != releases) {
            fprintf(stderr, "FAIL arena release grid %zu\n", i);
            ++failures;
        }
    }
    printf("test_heap_alignment: 3 grids, %d allocations, %d releases, %d failures\n",
           allocations, releases, failures);
    return failures ? 1 : 0;
}
