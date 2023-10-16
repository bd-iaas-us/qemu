#ifndef QEMU_DSA_H
#define QEMU_DSA_H

typedef void (*buffer_zero_dsa_completion_fn)(void *);

#ifdef CONFIG_DSA_OPT

#pragma GCC push_options
#pragma GCC target("enqcmd")
#pragma GCC target("movdir64b")

#include <linux/idxd.h>
#include "x86intrin.h"

#define DSA_BATCH_SIZE 128

enum dsa_task_status {
    DSA_TASK_READY = 0,
    DSA_TASK_PROCESSING,
    DSA_TASK_COMPLETION
};

struct dsa_buffer_zero_completion_context {
    //TODO: Add context structure to feed the callback fn.
};

struct buffer_zero_task {
    struct dsa_completion_record completion __attribute__((aligned(32)));
    struct dsa_hw_desc descriptor;
    buffer_zero_dsa_completion_fn completion_callback;
    struct dsa_buffer_zero_completion_context completion_context;
    enum dsa_task_status status;
};

struct buffer_zero_batch_task {
    struct dsa_hw_desc batch_descriptor;
    struct dsa_hw_desc descriptors[DSA_BATCH_SIZE] __attribute__((aligned(64)));
    struct dsa_completion_record batch_completion __attribute__((aligned(32)));
    struct dsa_completion_record completions[DSA_BATCH_SIZE] __attribute__((aligned(32)));
    struct dsa_device *device;
    buffer_zero_dsa_completion_fn completion_callback;
    struct dsa_buffer_zero_completion_context completion_contexts[DSA_BATCH_SIZE];
    enum dsa_task_status status;
};

void buffer_zero_task_init(struct buffer_zero_task *task);

// You must call init on a task before using it in a batch call.
void buffer_zero_batch_task_init(struct buffer_zero_batch_task *task);

// Returns 0 on success, or -1 on error.
// Right now the only error is if `count` is too big.
int buffer_is_zero_dsa_batch(struct buffer_zero_batch_task *batch_task,
                              const void **buf, size_t count,
                              size_t len, bool *result);

struct dsa_counters {
    uint64_t total_bytes_checked;
    uint64_t total_success_count;
    uint64_t total_batch_success_count;
    uint64_t total_bof_fail;
    uint64_t total_batch_fail;
    uint64_t total_batch_bof;
    uint64_t total_fallback_count;
    uint64_t top_retry_count;
};

extern struct dsa_counters dsa_counters; // Kind of awkward to have a public variable named `counters`

#endif

#endif