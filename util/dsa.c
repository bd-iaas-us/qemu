/*
 * Use Intel Data Streaming Accelerator to offload certain background
 * operations.
 *
 * Copyright (c) 2023 Hao Xiang <hao.xiang@bytedance.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "qemu/osdep.h"
#include "qemu/cutils.h"
#include "qemu/bswap.h"
#include "qemu/error-report.h"

#ifdef CONFIG_DSA_OPT

#pragma GCC push_options
#pragma GCC target("enqcmd")
#pragma GCC target("movdir64b")

#include <linux/idxd.h>
#include "x86intrin.h"

#define DSA_WQ_SIZE 4096
#define DSA_BATCH_SIZE 64

enum batch_task_status {
    BATCH_TASK_READY = 0,
    BATCH_TASK_PROCESSING,
    BATCH_TASK_COMPLETION
};

struct batch_buffer_zero_task {
    struct dsa_completion_record batch_completion __attribute__((aligned(32)));
    struct dsa_completion_record completions[DSA_BATCH_SIZE] __attribute__((aligned(32)));
    struct dsa_hw_desc batch_descriptor;
    struct dsa_hw_desc descriptors[DSA_BATCH_SIZE];
    enum batch_task_status status;
};

static bool use_simulation;
static uint64_t total_bytes_checked;
static uint64_t total_function_calls;
static uint64_t total_success_count;
static int max_retry_count;
static int top_retry_count;

static void *dsa_wq = MAP_FAILED;
static uint8_t zero_page_buffer[4096];
static bool dedicated_mode;
static int length_to_accel = 64;

static buffer_accel_fn buffer_zero_fallback;

/**
 * @brief This function opens a DSA device's work queue and
 *        maps the DSA device memory into the current process.
 *
 * @param dsa_wq_path A pointer to the DSA device work queue's file path.
 * @return A pointer to the mapped memory.
 */
static void *map_dsa_device(const char *dsa_wq_path)
{
    void *dsa_device;
    int fd;

    fd = open(dsa_wq_path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open %s failed with errno = %d.\n",
                dsa_wq_path, errno);
        return MAP_FAILED;
    }
    dsa_device = mmap(NULL, DSA_WQ_SIZE, PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    if (dsa_device == MAP_FAILED) {
        fprintf(stderr, "mmap failed with errno = %d.\n", errno);
        return MAP_FAILED;
    }
    return dsa_device;
}

/**
 * @brief Submits a DSA work item to the device work queue.
 *
 * @param wq A pointer to the DSA work queue's device memory.
 * @param descriptor A pointer to the DSA work item descriptor.
 * @return Zero if successful, non-zero otherwise.
 */
static int submit_wi(void *wq, void *descriptor)
{
    int retry = 0;

    _mm_sfence();

    if (dedicated_mode) {
        _movdir64b(dsa_wq, descriptor);
    } else {
        while (true) {
            if (_enqcmd(dsa_wq, descriptor) == 0) {
                break;
            }
            retry++;
            if (retry > max_retry_count) {
                fprintf(stderr, "Submit work retry %d times.\n", retry);
                exit(1);
            }
        }
    }

    return 0;
}

/**
 * @brief Poll for the DSA work item completion.
 *
 * @param completion A pointer to the DSA work item completion record.
 * @param opcode The DSA opcode.
 * @return Zero if successful, non-zero otherwise.
 */
static int poll_completion(struct dsa_completion_record *completion,
                           enum dsa_opcode opcode)
{
    int retry = 0;

    while (true) {
        if (completion->status != DSA_COMP_NONE) {
            /* TODO: Error handling here. */
            if (completion->status != DSA_COMP_SUCCESS &&
                completion->status != DSA_COMP_PAGE_FAULT_NOBOF) {
                fprintf(stderr, "DSA opcode %d failed with status = %d.\n",
                    opcode, completion->status);
                exit(1);
            } else {
                total_success_count++;
            }
            break;
        }
        retry++;
        if (retry > max_retry_count) {
            fprintf(stderr, "Wait for completion retry %d times.\n", retry);
            exit(1);
        }
        _mm_pause();
    }

    if (retry > top_retry_count) {
        top_retry_count = retry;
    }

    return 0;
}

static bool buffer_zero_dsa_simulation(const void *buf, size_t len)
{
    /* TODO: Handle page size greater than 4k. */
    if (len > sizeof(zero_page_buffer)) {
        fprintf(stderr, "Page size greater than %lu is not supported by DSA "
                        "buffer zero checking.\n", sizeof(zero_page_buffer));
        exit(1);
    }

    total_bytes_checked += len;
    total_function_calls++;

    return memcmp(buf, zero_page_buffer, len) == 0;
}

/**
 * @brief Initializes a buffer zero batch task.
 *
 * @param task A pointer to the batch task to initialize.
 */
__attribute__((unused))
static void batch_buffer_zero_task_init(struct batch_buffer_zero_task *task)
{
    task->batch_completion.status = DSA_COMP_NONE;
    task->batch_descriptor.completion_addr = (uint64_t)&task->batch_completion;
    task->batch_descriptor.xfer_size = 0;
    task->batch_descriptor.desc_count = 0;
    task->batch_descriptor.opcode = DSA_OPCODE_BATCH;
    task->batch_descriptor.flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
    task->batch_descriptor.src_addr = (uintptr_t)task->descriptors;
    task->batch_descriptor.dst_addr = 0;
    task->status = BATCH_TASK_READY;
}

/**
 * @brief Initializes a buffer zero task.
 *
 * @param descriptor A pointer to the DSA task descriptor.
 * @param completion A pointer to the DSA task completion record.
 * @param buf A pointer to the memory buffer to check for zero.
 * @param len The length of the buffer.
 */
static void
buffer_zero_task_init(struct dsa_hw_desc *descriptor,
                      struct dsa_completion_record *completion,
                      const void *buf, size_t len)
{
    /* TODO: Handle page size greater than 4k. */
    if (len > sizeof(zero_page_buffer)) {
        fprintf(stderr, "Page size greater than %lu is not supported by DSA "
                        "buffer zero checking.\n", sizeof(zero_page_buffer));
        exit(1);
    }

    total_bytes_checked += len;

    //memset(&completion, 0, sizeof(completion));
    //memset(&descriptor, 0, sizeof(descriptor));

    descriptor->opcode = DSA_OPCODE_COMPARE;
    descriptor->flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
    descriptor->xfer_size = len;
    descriptor->src_addr = (uintptr_t)buf;
    descriptor->dst_addr = (uintptr_t)zero_page_buffer;
    completion->status = 0;
    descriptor->completion_addr = (uint64_t)&completion;
}

static void page_in(const void *buf)
{
    uint8_t test_byte;

    /*
     * TODO: Find a better solution. DSA device can encounter page
     * fault during the memory comparison operatio. Block on page
     * fault is turned off for better performance. This temporary
     * solution reads the first byte of the memory buffer in order
     * to cause a CPU page fault so that DSA device won't hit that
     * later.
     */
    test_byte = ((uint8_t *)buf)[0];
    ((uint8_t *)buf)[0] = test_byte;
}

/**
 * @brief Sends a memory comparison task to a DSA device and wait
 *        for completion.
 *
 * @param buf A pointer to the memory buffer for comparison.
 * @param len Length of the memory buffer for comparison.
 * @return true if the memory buffer is all zero, false otherwise.
 */
static bool buffer_zero_dsa(const void *buf, size_t len)
{
    struct dsa_completion_record completion __attribute__((aligned(32)));
    struct dsa_hw_desc descriptor;

    buffer_zero_task_init(&descriptor, &completion, buf, len);

    total_function_calls++;

    page_in(buf);

    submit_wi(dsa_wq, &descriptor);
    poll_completion(&completion, DSA_OPCODE_COMPARE);

    if (completion.status == DSA_COMP_SUCCESS) {
        return completion.result == 0;
    }

    /*
     * DSA was able to partially complete the operation. Check the
     * result. If we already know this is not a zero page, we can
     * return now.
     */
    if (completion.bytes_completed != 0 && completion.result != 0) {
        return false;
    }

    /* Let's fallback to use CPU to complete it. */
    return buffer_zero_fallback((uint8_t *)buf + completion.bytes_completed,
                                len - completion.bytes_completed);
}

/**
 * @brief Add a buffer zero task to the batch task.
 *
 * @param batch_task A pointer to the batch task.
 * @param buf A pointer to the memory buffer to check for zero.
 * @param len The length of the buffer.
 *
 * @return true if successful, otherwise false.
 */
__attribute__((unused))
static bool
buffer_zero_dsa_batch_add_task(struct batch_buffer_zero_task *batch_task,
                               const void *buf, size_t len)
{
    int desc_count;

    assert(batch_task->status == BATCH_TASK_READY);

    if (batch_task->batch_descriptor.desc_count >= DSA_BATCH_SIZE)
        return false;

    desc_count = batch_task->batch_descriptor.desc_count;
    buffer_zero_task_init(&batch_task->descriptors[desc_count],
                          &batch_task->completions[desc_count],
                          buf, len);

    batch_task->batch_descriptor.xfer_size++;
    batch_task->batch_descriptor.desc_count++;

    return true;
}

/**
 * @brief Sends a memory comparison batch task to a DSA device and wait
 *        for completion.
 *
 * @param batch_task The batch task to be submitted to DSA device.
 */
__attribute__((unused))
static void buffer_zero_dsa_batch(struct batch_buffer_zero_task *batch_task)
{
    assert(batch_task->batch_descriptor.desc_count <= DSA_BATCH_SIZE);
    assert(batch_task->status == BATCH_TASK_READY);

    for (int i = 0; i < batch_task->batch_descriptor.desc_count; i++) {
        page_in((void*)batch_task->descriptors[i].src_addr);
    }

    batch_task->status = BATCH_TASK_PROCESSING;

    submit_wi(dsa_wq, &batch_task->batch_descriptor);
    poll_completion(&batch_task->batch_completion, DSA_OPCODE_BATCH);

    //TODO: Poll completion on all individual tasks in the batch.

    batch_task->status = BATCH_TASK_COMPLETION;
}

/**
 * @brief Check if DSA devices are enabled in the current system
 *        and set DSA offloading for zero page checking operation.
 *        This function is called during QEMU initialization.
 *
 * @param dsa_path A pointer to the DSA device's work queue file path.
 * @return int Zero if successful, non-zero otherwise.
 */
int configure_dsa(const char *dsa_path)
{
    dedicated_mode = false;
    use_simulation = false;
    max_retry_count = 3000;
    total_bytes_checked = 0;
    total_function_calls = 0;
    total_success_count = 0;

    memset(zero_page_buffer, 0, sizeof(zero_page_buffer));

    dsa_wq = map_dsa_device(dsa_path);
    if (dsa_wq == MAP_FAILED) {
        fprintf(stderr, "map_dsa_device failed MAP_FAILED, "
                "using simulation.\n");
        return -1;
    }

    if (use_simulation)
        set_accel(buffer_zero_dsa_simulation, length_to_accel);
    else {
        set_accel(buffer_zero_dsa, length_to_accel);
        get_fallback_accel(&buffer_zero_fallback);
    }

    return 0;
}

/**
 * @brief Clean up system resources created for DSA offloading.
 *        This function is called during QEMU process teardown.
 *
 */
void dsa_cleanup(void)
{
    if (dsa_wq != MAP_FAILED) {
        munmap(dsa_wq, DSA_WQ_SIZE);
    }
}

#else

int configure_dsa(const char *dsa_path)
{
    fprintf(stderr, "Intel Data Streaming Accelerator is not supported "
                    "on this platform.\n");
    return -1;
}

void dsa_cleanup(void) {}

#endif
