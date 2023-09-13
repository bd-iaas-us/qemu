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
#include "qemu/queue.h"
#include "qemu/lockable.h"
#include "qemu/cutils.h"
#include "qemu/dsa.h"
#include "qemu/bswap.h"
#include "qemu/error-report.h"
#include "qemu/rcu.h"

#ifdef CONFIG_DSA_OPT

#pragma GCC push_options
#pragma GCC target("enqcmd")
#pragma GCC target("movdir64b")

#include <linux/idxd.h>
#include "x86intrin.h"

#define DSA_WQ_SIZE 4096
#define DSA_COMPLETION_THREAD "dsa_completion_thread"

enum dsa_task_type {
    DSA_TASK = 0,
    DSA_BATCH_TASK
};

typedef struct dsa_task_entry {
    QSIMPLEQ_ENTRY(dsa_task_entry) entry;
    enum dsa_task_type task_type;
    union {
        struct buffer_zero_task *task;
        struct buffer_zero_batch_task *batch_task;
    };
} dsa_task_entry;

typedef QSIMPLEQ_HEAD(dsa_task_queue, dsa_task_entry) dsa_task_queue;

struct dsa_device {
    bool running;
    void *work_queue;
    QemuMutex task_queue_lock;
    QemuCond task_queue_cond;
    dsa_task_queue task_queue;
};

struct dsa_completion_thread {
    bool stopping;
    bool running;
    QemuThread thread;
    int thread_id;
    QemuSemaphore init_done_sem;
    struct dsa_device *dsa_device_instance;
};

static uint64_t total_bytes_checked;
static uint64_t total_function_calls;
static uint64_t total_success_count;
static int max_retry_count;
static int top_retry_count;

static bool dedicated_mode;
static int length_to_accel = 64;

static buffer_accel_fn buffer_zero_fallback;

static struct dsa_device dsa_device_instance;

/**
 * @brief Initializes a DSA device structure.
 * 
 * @param instance A pointer to the DSA device.
 * @param work_queue  A pointer to the DSA work queue.
 */
static void
dsa_device_init(struct dsa_device *instance,
                void *dsa_work_queue)
{
    instance->running = true;
    instance->work_queue = dsa_work_queue;
    qemu_mutex_init(&instance->task_queue_lock);
    qemu_cond_init(&instance->task_queue_cond);
    QSIMPLEQ_INIT(&instance->task_queue);
}

/**
 * @brief Stops a DSA device instance.
 * 
 * @param instance A pointer to the DSA device to stop.
 */
static void
dsa_device_stop(struct dsa_device *instance)
{
    qemu_mutex_lock(&instance->task_queue_lock);
    instance->running = false;
    qemu_cond_signal(&instance->task_queue_cond);
    qemu_mutex_unlock(&instance->task_queue_lock);
}

/**
 * @brief Cleans up a DSA device structure.
 * 
 * @param instance A pointer to the DSA device to cleanup.
 */
static void
dsa_device_cleanup(struct dsa_device *instance)
{
    if (instance->work_queue != MAP_FAILED) {
        munmap(instance->work_queue, DSA_WQ_SIZE);
    }
}

/**
 * @brief Adds a task to the DSA task queue.
 * 
 * @param device_instance A pointer to the DSA device.
 * @param type The DSA task type.
 * @param context A pointer to the DSA task to enqueue.
 *
 * @return int Zero if successful, otherwise a proper error code.
 */
static int
dsa_task_enqueue(struct dsa_device *device_instance,
                 enum dsa_task_type type,
                 void *context)
{
    dsa_task_queue *task_queue = &device_instance->task_queue;
    QemuMutex *task_queue_lock = &device_instance->task_queue_lock;
    QemuCond *task_queue_cond = &device_instance->task_queue_cond;

    // TODO: Use pre-allocated lookaside buffer instead.
    dsa_task_entry *task_entry = malloc(sizeof(dsa_task_entry));
    bool notify = false;

    task_entry->task_type = type;
    if (type == DSA_TASK)
        task_entry->task = context;
    else {
        assert(type == DSA_BATCH_TASK);
        task_entry->batch_task = context;
    }

    qemu_mutex_lock(task_queue_lock);

    // The queue is empty. This enqueue operation is a 0->1 transition.
    if (QSIMPLEQ_EMPTY(task_queue))
        notify = true;

    QSIMPLEQ_INSERT_TAIL(task_queue, task_entry, entry);

    // We need to notify the waiter for 0->1 transitions.
    if (notify)
        qemu_cond_signal(task_queue_cond);

    qemu_mutex_unlock(task_queue_lock);

    return 0;
}

/**
 * @brief Takes a DSA task out of the task queue.
 * 
 * @param device_instance A pointer to the DSA device.
 * @return dsa_task_entry* The DSA task being dequeued.
 */
static dsa_task_entry *
dsa_task_dequeue(struct dsa_device *device_instance)
{
    dsa_task_entry *task_entry = NULL;
    dsa_task_queue *task_queue = &device_instance->task_queue;
    QemuMutex *task_queue_lock = &device_instance->task_queue_lock;
    QemuCond *task_queue_cond = &device_instance->task_queue_cond;

    qemu_mutex_lock(task_queue_lock);

    while (true) {
        if (!device_instance->running)
            goto exit;
        task_entry = QSIMPLEQ_FIRST(task_queue);
        if (task_entry != NULL) {
            break;
        }
        qemu_cond_wait(task_queue_cond, task_queue_lock);
    }
        
    QSIMPLEQ_REMOVE_HEAD(task_queue, entry);

exit:
    qemu_mutex_unlock(task_queue_lock);
    return task_entry;
}

/**
 * @brief This function opens a DSA device's work queue and
 *        maps the DSA device memory into the current process.
 *
 * @param dsa_wq_path A pointer to the DSA device work queue's file path.
 * @return A pointer to the mapped memory.
 */
static void *
map_dsa_device(const char *dsa_wq_path)
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
static int
submit_wi_int(void *wq, struct dsa_hw_desc *descriptor)
{
    int retry = 0;

    _mm_sfence();

    if (dedicated_mode) {
        _movdir64b(wq, descriptor);
    } else {
        while (true) {
            if (_enqcmd(wq, descriptor) == 0) {
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
 * @brief Synchronously submits a DSA work item to the
 *        device work queue.
 * 
 * @param wq A pointer to the DSA worjk queue's device memory.
 * @param descriptor A pointer to the DSA work item descriptor.
 * @return int Zero if successful, non-zero otherwise.
 */
static int
submit_wi(void *wq, struct dsa_hw_desc *descriptor)
{
    return submit_wi_int(wq, descriptor);
}

/**
 * @brief Asynchronously submits a DSA work item to the
 *        device work queue.
 * 
 * @param device_instance A pointer to the DSA device instance.
 * @param task A pointer to the buffer zero task.
 * @return int Zero if successful, non-zero otherwise.
 */
static int
submit_wi_async(struct dsa_device *device_instance,
                struct buffer_zero_task *task)
{
    int ret;

    task->status = DSA_TASK_PROCESSING;

    ret = submit_wi_int(device_instance->work_queue,
                        &task->descriptor);
    if (ret != 0)
        return ret;

    return dsa_task_enqueue(device_instance, DSA_TASK, task);
}

/**
 * @brief Asynchronously submits a DSA batch work item to the
 *        device work queue.
 * 
 * @param device_instance A pointer to the DSA device instance.
 * @param batch_task A pointer to the batch buffer zero task.
 * @return int Zero if successful, non-zero otherwise.
 */
static int
submit_batch_wi_async(struct dsa_device *device_instance,
                      struct buffer_zero_batch_task *batch_task)
{
    int ret;

    batch_task->status = DSA_TASK_PROCESSING;

    ret = submit_wi_int(device_instance->work_queue,
                        &batch_task->batch_descriptor);
    if (ret != 0)
        return ret;

    return dsa_task_enqueue(device_instance, DSA_BATCH_TASK, batch_task);
}

/**
 * @brief Poll for the DSA work item completion.
 *
 * @param completion A pointer to the DSA work item completion record.
 * @param opcode The DSA opcode.
 * @return Zero if successful, non-zero otherwise.
 */
static int
poll_completion(struct dsa_completion_record *completion,
                enum dsa_opcode opcode)
{
    int retry = 0;

    while (true) {
        if (completion->status != DSA_COMP_NONE) {
            /* TODO: Error handling here. */
            if (completion->status != DSA_COMP_SUCCESS &&
                completion->status != DSA_COMP_PAGE_FAULT_NOBOF &&
                completion->status != DSA_COMP_BATCH_FAIL &&
	            completion->status != DSA_COMP_BATCH_PAGE_FAULT) {
                fprintf(stderr, "DSA opcode %d failed with status = %d.\n",
                    opcode, completion->status);
                exit(1);
            } else {
                total_success_count++;
                //fprintf(stderr, "poll_completion retried %d times.\n", retry);
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

/**
 * @brief Handles an asynchronous DSA task completion.
 * 
 * @param task A pointer to the buffer zero task structure.
 */
static void
dsa_task_complete(struct buffer_zero_task *task)
{
    task->status = DSA_TASK_COMPLETION;
    task->completion_callback(&task->completion_context);
}

/**
 * @brief Handles an asynchronous DSA batch task completion.
 * 
 * @param task A pointer to the batch buffer zero task structure.
 */
static void
dsa_batch_task_complete(struct buffer_zero_batch_task *batch_task)
{
    batch_task->status = DSA_TASK_COMPLETION;
    for (int i = 0; i < batch_task->batch_descriptor.desc_count; i++)
        batch_task->completion_callback(&batch_task->completion_contexts[i]);
}

/**
 * @brief The function entry point called by a dedicated DSA
 *        work item completion thread. 
 * 
 * @param opaque A pointer to the thread context.
 * @return void* 
 */
static void *
dsa_completion_loop(void *opaque)
{
    struct dsa_completion_thread *thread_context =
        (struct dsa_completion_thread *)opaque;
    dsa_task_entry *task_entry;
    struct dsa_device *dsa_device_instance = thread_context->dsa_device_instance;

    rcu_register_thread();

    thread_context->thread_id = qemu_get_thread_id();
    qemu_sem_post(&thread_context->init_done_sem);

    while (thread_context->running) {
        task_entry = dsa_task_dequeue(dsa_device_instance);
        assert(task_entry != NULL || !dsa_device_instance->running);
        if (!dsa_device_instance->running) {
            assert(!thread_context->running);
            break;
        }
        if (task_entry->task_type == DSA_TASK) {
            poll_completion(&task_entry->task->completion,
                            task_entry->task->descriptor.opcode);
            dsa_task_complete(task_entry->task);
        } else {
            assert(task_entry->task_type == DSA_BATCH_TASK);
            poll_completion(&task_entry->batch_task->batch_completion,
                            task_entry->batch_task->batch_descriptor.opcode);
            dsa_batch_task_complete(task_entry->batch_task);
        }
        // TODO: Use pre-allocated lookaside buffer instead.
        free(task_entry);
    }

    rcu_unregister_thread();
    return NULL;
}

/**
 * @brief Initializes a DSA completion thread.
 * 
 * @param completion_thread A pointer to the completion thread context.
 */
__attribute__((unused))
static void
dsa_completion_thread_init(
    struct dsa_completion_thread *completion_thread)
{
    completion_thread->stopping = false;
    completion_thread->running = true;
    completion_thread->dsa_device_instance = &dsa_device_instance;
    completion_thread->thread_id = -1;
    qemu_sem_init(&completion_thread->init_done_sem, 0);

    qemu_thread_create(&completion_thread->thread,
                       DSA_COMPLETION_THREAD,
                       dsa_completion_loop,
                       completion_thread,
                       QEMU_THREAD_JOINABLE);

    /* Wait for initialization to complete */
    while (completion_thread->thread_id == -1) {
        qemu_sem_wait(&completion_thread->init_done_sem);
    }
}

/**
 * @brief Stops the DSA completion thread.
 * 
 * @param opaque A pointer to the thread context.
 */
__attribute__((unused))
static void
dsa_completion_thread_stop(void *opaque)
{
    struct dsa_completion_thread *thread_context =
        (struct dsa_completion_thread *)opaque;

    thread_context->stopping = true;
    thread_context->running = false;

    dsa_device_stop(thread_context->dsa_device_instance);

    qemu_thread_join(&thread_context->thread);    
}

/**
 * @brief Initializes a buffer zero task.
 *
 * @param task A pointer to the buffer_zero_task structure.
 * @param completion_callback The callback function for DSA completion.
 * @param buf A pointer to the memory buffer to check for zero.
 * @param len The length of the buffer.
 */
static void
buffer_zero_task_init_int(struct dsa_hw_desc *descriptor,
                          struct dsa_completion_record *completion)
{
    descriptor->opcode = DSA_OPCODE_COMPVAL;
    descriptor->flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
    descriptor->comp_pattern = (uint64_t)0;
    descriptor->completion_addr = (uint64_t)completion;
}

/**
 * @brief Initializes a buffer zero task.
 * 
 * @param task A pointer to the buffer zero task.
 * @param completion_callback The DSA task completion callback function.
 * @param buf A pointer to the memory buffer to check for zero.
 * @param len The length of the memory buffer.
 */
void
buffer_zero_task_init(struct buffer_zero_task *task)
{
    memset(task, 0, sizeof(*task));

    buffer_zero_task_init_int(&task->descriptor,
                              &task->completion);
}

/**
 * @brief Initializes a buffer zero batch task.
 *
 * @param task A pointer to the batch task to initialize.
 */
void
buffer_zero_batch_task_init(struct buffer_zero_batch_task *task)
{
    memset(task, 0, sizeof(*task));

    task->batch_completion.status = DSA_COMP_NONE;
    task->batch_descriptor.completion_addr = (uint64_t)&task->batch_completion;
    // TODO: Ensure that we never send a batch with count <= 1
    task->batch_descriptor.desc_count = 0;
    task->batch_descriptor.opcode = DSA_OPCODE_BATCH;
    task->batch_descriptor.flags = IDXD_OP_FLAG_RCR | IDXD_OP_FLAG_CRAV;
    task->batch_descriptor.desc_list_addr = (uintptr_t)task->descriptors;
    task->status = DSA_TASK_READY;

    for (int i = 0; i < DSA_BATCH_SIZE; i++) {
        buffer_zero_task_init_int(&task->descriptors[i],
                                  &task->completions[i]);
    }
}

static void
buffer_zero_task_set_int(struct dsa_hw_desc *descriptor,
                         const void *buf,
                         size_t len)
{
    struct dsa_completion_record *completion = 
        (struct dsa_completion_record *)descriptor->completion_addr;

    total_bytes_checked += len;

    descriptor->xfer_size = len;
    descriptor->src_addr = (uintptr_t)buf;
    completion->status = 0;
}

static void
buffer_zero_task_set(struct buffer_zero_task *task,
                     const void *buf,
                     size_t len)
{
    buffer_zero_task_set_int(&task->descriptor, buf, len);
}

/**
 * @brief Initializes a buffer zero task inside a batch task.
 * 
 * @param batch_task A pointer to the batch buffer zero task.
 * @param index The index to the buffer zero task inside the batch.  
 * @param buf The memory buffer to check for zero.
 * @param len The length of the buffer.
 */
static void
buffer_zero_task_in_batch_set(struct buffer_zero_batch_task *batch_task,
                              int index, const void *buf, size_t len)
{
    buffer_zero_task_set_int(&batch_task->descriptors[index],
                             buf, len);
}

static void
buffer_zero_batch_task_set(struct buffer_zero_batch_task *batch_task,
                           const void **buf, size_t count, size_t len)
{
    assert(count > 1);
    assert(count <= DSA_BATCH_SIZE);

    for (int i = 0; i < count; i++) {
        buffer_zero_task_set_int(&batch_task->descriptors[i], buf[i], len);
    }
    batch_task->batch_descriptor.desc_count = count;
}

/**
 * @brief Forces the OS to page-in the memory buffer by writing
 *        to the first byte in the buffer.
 * 
 * @param buf The memory buffer to be paged in.
 */
static void
page_in(const void *buf)
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
static bool
buffer_zero_dsa(const void *buf, size_t len)
{
    struct buffer_zero_task task;
    struct dsa_completion_record *completion = &task.completion;

    buffer_zero_task_init(&task);
    buffer_zero_task_set(&task, buf, len);

    total_function_calls++;

    page_in(buf);

    submit_wi(dsa_device_instance.work_queue, &task.descriptor);
    poll_completion(completion, task.descriptor.opcode);

    if (completion->status == DSA_COMP_SUCCESS) {
        return completion->result == 0;
    }

    /*
     * DSA was able to partially complete the operation. Check the
     * result. If we already know this is not a zero page, we can
     * return now.
     */
    if (completion->bytes_completed != 0 && completion->result != 0) {
        return false;
    }

    /* Let's fallback to use CPU to complete it. */
    return buffer_zero_fallback((uint8_t *)buf + completion->bytes_completed,
                                len - completion->bytes_completed);
}

static void
buffer_zero_dsa_batch(struct buffer_zero_batch_task *batch_task,
                      const void **buf, size_t count, size_t len, 
                      bool *result)
{
    struct dsa_completion_record *completion;

    assert(count <= DSA_BATCH_SIZE);

    buffer_zero_batch_task_set(batch_task, buf, count, len);

    total_function_calls++;

    //for (int i = 0; i < count; i++) {
    //    page_in(buf[i]);
    //}

    submit_wi(dsa_device_instance.work_queue, &batch_task->batch_descriptor);
    poll_completion(&batch_task->batch_completion, batch_task->batch_descriptor.opcode);

    for (int i = 0; i < count; i++) {

        completion = &batch_task->completions[i];

        if (completion->status == DSA_COMP_SUCCESS) {
            result[i] = (completion->result == 0);
            continue;
        }

        /*
        * DSA was able to partially complete the operation. Check the
        * result. If we already know this is not a zero page, we can
        * return now.
        */
        if (completion->bytes_completed != 0 && completion->result != 0) {
            result[i] = false;
            continue;
        }

        /* Let's fallback to use CPU to complete it. */
        result[i] = buffer_zero_fallback((uint8_t *)buf[i] + completion->bytes_completed,
                                         len - completion->bytes_completed);
    }
}

/**
 * @brief Asychronously perform a buffer zero DSA operation.
 * 
 * @param buf A pointer to the memory buffer.
 * @param len The length of the memory buffer.
 * @param completion_fn The DSA task completion callback function.
 * @param completion_context The DSA task completion callback context.
 * @return int Zero if successful, otherwise an appropriate error code.
 */
__attribute__((unused))
static int
buffer_zero_dsa_async(const void *buf, size_t len,
                      buffer_zero_dsa_completion_fn completion_fn,
                      void *completion_context)
{
    struct buffer_zero_task *task = 
        aligned_alloc(32, sizeof(struct buffer_zero_task));

    buffer_zero_task_init(task);
    buffer_zero_task_set(task, buf, len);

    total_function_calls++;

    page_in(buf);

    return submit_wi_async(dsa_device_instance.work_queue, task);
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
buffer_zero_dsa_batch_add_task(struct buffer_zero_batch_task *batch_task,
                               const void *buf, size_t len)
{
    int desc_count;

    assert(batch_task->status == DSA_TASK_READY);

    if (batch_task->batch_descriptor.desc_count >= DSA_BATCH_SIZE)
        return false;

    desc_count = batch_task->batch_descriptor.desc_count;
    buffer_zero_task_in_batch_set(batch_task, desc_count, buf, len);

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
static int
buffer_zero_dsa_batch_async(struct buffer_zero_batch_task *batch_task)
{
    assert(batch_task->batch_descriptor.desc_count <= DSA_BATCH_SIZE);
    assert(batch_task->status == DSA_TASK_READY);

    for (int i = 0; i < batch_task->batch_descriptor.desc_count; i++) {
        page_in((void*)batch_task->descriptors[i].src_addr);
    }

    return submit_batch_wi_async(&dsa_device_instance, batch_task);
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
    void *dsa_wq = MAP_FAILED;
    dedicated_mode = true;
    max_retry_count = 100000;
    total_bytes_checked = 0;
    total_function_calls = 0;
    total_success_count = 0;

    dsa_wq = map_dsa_device(dsa_path);
    if (dsa_wq == MAP_FAILED) {
        fprintf(stderr, "map_dsa_device failed MAP_FAILED, "
                "using simulation.\n");
        return -1;
    }

    set_accel(buffer_zero_dsa, length_to_accel);
    get_fallback_accel(&buffer_zero_fallback);

    dsa_device_init(&dsa_device_instance, dsa_wq);

    return 0;
}

/**
 * @brief Clean up system resources created for DSA offloading.
 *        This function is called during QEMU process teardown.
 *
 */
void dsa_cleanup(void)
{
    dsa_device_cleanup(&dsa_device_instance);
}

void buffer_is_zero_dsa_batch(struct buffer_zero_batch_task *batch_task,
                              const void **buf, size_t count,
                              size_t len, bool *result)
{
    if (count == 0) {
        return;
    }

    assert(len != 0);

    if (count == 1) {
        // DSA doesn't take batch operation with only 1 task.
        buffer_zero_dsa(buf, len);
    } else {
        buffer_zero_dsa_batch(batch_task, buf, count, len, result);
    }
}

#else

void buffer_is_zero_dsa_batch(struct buffer_zero_batch_task *batch_task,
                              const void **buf, size_t count,
                              size_t len, bool *result)
{
    exit(1);
}

int configure_dsa(const char *dsa_path)
{
    fprintf(stderr, "Intel Data Streaming Accelerator is not supported "
                    "on this platform.\n");
    return -1;
}

void dsa_cleanup(void) {}

#endif
