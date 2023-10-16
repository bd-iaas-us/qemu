#ifndef QEMU_DSA_H
#define QEMU_DSA_H

#include "qemu/thread.h"
#include "qemu/queue.h"

typedef void (*buffer_zero_dsa_completion_fn)(void *);

#define DSA_BATCH_SIZE 128

#ifdef CONFIG_DSA_OPT

#pragma GCC push_options
#pragma GCC target("enqcmd")
#pragma GCC target("movdir64b")

#include <linux/idxd.h>
#include "x86intrin.h"

enum dsa_task_type {
    DSA_TASK = 0,
    DSA_BATCH_TASK
};

enum dsa_task_status {
    DSA_TASK_READY = 0,
    DSA_TASK_PROCESSING,
    DSA_TASK_COMPLETION
};

typedef struct buffer_zero_batch_task {
    struct dsa_hw_desc batch_descriptor;
    //struct dsa_hw_desc descriptors[DSA_BATCH_SIZE] __attribute__((aligned(64)));
    struct dsa_hw_desc *descriptors;
    struct dsa_completion_record batch_completion __attribute__((aligned(32)));
    //struct dsa_completion_record completions[DSA_BATCH_SIZE] __attribute__((aligned(32)));
    struct dsa_completion_record *completions;
    struct dsa_device_group *group;
    struct dsa_device *device;
    buffer_zero_dsa_completion_fn completion_callback;
    QemuSemaphore sem_task_complete;
    enum dsa_task_type task_type;
    enum dsa_task_status status;
    //bool results[DSA_BATCH_SIZE];
    bool *results;
    int batch_size;
    QSIMPLEQ_ENTRY(buffer_zero_batch_task) entry;
} buffer_zero_batch_task;

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

#else 

struct buffer_zero_batch_task {
    bool results[DSA_BATCH_SIZE];
};

struct dsa_counters {

};

#endif 

/**
 * @brief Get a read-only reference to DSA counters.
 * 
 * @return Read-only DSA counters.
 */
struct dsa_counters* dsa_get_counters(void);

/**
 * @brief Check if DSA is running.
 * 
 * @return True if DSA is running, otherwise false. 
 */
bool dsa_is_running(void);

/**
 * @brief Initializes a buffer zero batch task.
 *
 * @param task A pointer to the batch task to initialize.
 * @param batch_size The number of DSA tasks in the batch.
 */
void buffer_zero_batch_task_init(struct buffer_zero_batch_task *task,
                                 int batch_size);

/**
 * @brief Performs the proper cleanup on a DSA batch task.
 * 
 * @param task A pointer to the batch task to cleanup.
 */
void buffer_zero_batch_task_destroy(struct buffer_zero_batch_task *task);

/**
 * @brief Performs buffer zero comparison on a DSA batch task synchronously.
 * 
 * @param batch_task A pointer to the batch task.
 * @param buf An array of memory buffers.
 * @param count The number of buffers in the array.
 * @param len The buffer length.
 */
int buffer_is_zero_dsa_batch(struct buffer_zero_batch_task *batch_task,
                             const void **buf, size_t count,
                             size_t len);

/**
 * @brief Performs buffer zero comparison on a DSA batch task asynchronously.
 * 
 * @param batch_task A pointer to the batch task.
 * @param buf An array of memory buffers.
 * @param count The number of buffers in the array.
 * @param len The buffer length.
 *
 * @return Zero if successful, otherwise non-zero.
 */
int
buffer_is_zero_dsa_batch_async(struct buffer_zero_batch_task *batch_task,
                               const void **buf, size_t count, size_t len);

#endif