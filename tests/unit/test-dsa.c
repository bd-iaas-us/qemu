/*
 * Test DSA functions.
 *
 * Copyright (c) 2023 Hao Xiang <hao.xiang@bytedance.com>
 * Copyright (c) 2023 Bryan Zhang <bryan.zhang@bytedance.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
#include "qemu/osdep.h"
#include "qemu/host-utils.h"

#include "qemu/cutils.h"
#include "qemu/memalign.h"
#include "qemu/dsa.h"

// TODO Make these not-hardcoded.
static const char *path[] = {"/dev/dsa/wq4.0", "/dev/dsa/wq4.1"};
static const int num_devices = 2;

static struct buffer_zero_batch_task batch_task __attribute__((aligned(64)));

// TODO Communicate that DSA must be configured to support this batch size.
// TODO Alternatively, poke the DSA device to figure out batch size.
static int batch_size = 128;
static int page_size = 4096;

// A helper for running a single task and checking for correctness.
static void do_single_task(bool async)
{
    buffer_zero_batch_task_init(&batch_task, batch_size);
    char buf[page_size];
    char* ptr = buf;

    if (async) {
        buffer_is_zero_dsa_batch_async(&batch_task,
                                       (const void**) &ptr,
                                       1,
                                       page_size);
    } else {
        buffer_is_zero_dsa_batch(&batch_task,
                            (const void**) &ptr,
                            1,
                            page_size);
    }
    g_assert(batch_task.results[0] == buffer_is_zero(buf, page_size));
}

static void test_single_zero(bool async)
{
    g_assert(!dsa_configure(path, 1));
    dsa_start();

    buffer_zero_batch_task_init(&batch_task, batch_size);

    char buf[page_size];
    char* ptr = buf;

    memset(buf, 0x0, page_size);
    if (async) {
        buffer_is_zero_dsa_batch_async(&batch_task,
                                       (const void**) &ptr,
                                       1,
                                       page_size);
    } else {
        buffer_is_zero_dsa_batch(&batch_task,
                                (const void**) &ptr,
                                1,
                                page_size);
    }
    g_assert(batch_task.results[0]);
    
    dsa_cleanup();
}

static void test_single_zero_async(void)
{
    test_single_zero(true);
}

static void test_single_zero_sync(void)
{
    test_single_zero(false);
}

static void test_single_nonzero(bool async)
{
    g_assert(!dsa_configure(path, 1));
    dsa_start();

    buffer_zero_batch_task_init(&batch_task, batch_size);

    char buf[page_size];
    char* ptr = buf;

    memset(buf, 0x1, page_size);
    if (async) {
        buffer_is_zero_dsa_batch_async(&batch_task,
                                       (const void**) &ptr,
                                       1,
                                       page_size);
    } else {
        buffer_is_zero_dsa_batch(&batch_task,
                                (const void**) &ptr,
                                1,
                                page_size);
    }
    g_assert(!batch_task.results[0]);
    
    dsa_cleanup(); 
}

static void test_single_nonzero_async(void)
{
    test_single_nonzero(true);
}

static void test_single_nonzero_sync(void)
{
    test_single_nonzero(false);
}

// count == 0 should return quickly without calling into DSA.
static void test_zero_count_async(void)
{
    char buf[page_size];
    buffer_is_zero_dsa_batch_async(&batch_task,
                             (const void **) &buf,
                             0,
                             page_size);
}

static void test_zero_count_sync(void)
{
    char buf[page_size];
    buffer_is_zero_dsa_batch(&batch_task,
                             (const void **) &buf,
                             0,
                             page_size);
}

static void test_null_task_async(void)
{
    if (g_test_subprocess()) {
        g_assert(!dsa_configure(path, 1));

        char buf[page_size * batch_size];
        char *addrs[batch_size];
        for (int i = 0; i < batch_size; i++) {
            addrs[i] = buf + (page_size * i);
        }

        buffer_is_zero_dsa_batch_async(NULL, (const void**) addrs, batch_size,
                                 page_size);
    } else {
        g_test_trap_subprocess(NULL, 0, 0);
        g_test_trap_assert_failed();
    }
}

static void test_null_task_sync(void)
{
    if (g_test_subprocess()) {
        g_assert(!dsa_configure(path, 1));

        char buf[page_size * batch_size];
        char *addrs[batch_size];
        for (int i = 0; i < batch_size; i++) {
            addrs[i] = buf + (page_size * i);
        }

        buffer_is_zero_dsa_batch(NULL, (const void**) addrs, batch_size,
                                 page_size);
    } else {
        g_test_trap_subprocess(NULL, 0, 0);
        g_test_trap_assert_failed();
    }
}

static void test_oversized_batch(bool async)
{
    g_assert(!dsa_configure(path, 1));
    dsa_start();

    buffer_zero_batch_task_init(&batch_task, batch_size);

    int oversized_batch_size = batch_size + 1;
    char buf[page_size * oversized_batch_size];
    char *addrs[batch_size];
    for (int i = 0; i < oversized_batch_size; i++) {
        addrs[i] = buf + (page_size * i);
    }

    int ret;
    if (async) {
        ret = buffer_is_zero_dsa_batch_async(&batch_task,
                                            (const void**) addrs,
                                            oversized_batch_size,
                                            page_size);
    } else {
        ret = buffer_is_zero_dsa_batch(&batch_task,
                                       (const void**) addrs,
                                       oversized_batch_size,
                                       page_size);
    }
    g_assert(ret != 0);

    dsa_cleanup();
}

static void test_oversized_batch_async(void)
{
    test_oversized_batch(true);
}

static void test_oversized_batch_sync(void)
{
    test_oversized_batch(false);
}

static void test_zero_len_async(void)
{
    if (g_test_subprocess()) {
        g_assert(!dsa_configure(path, 1));

        buffer_zero_batch_task_init(&batch_task, batch_size);

        char buf[page_size];

        buffer_is_zero_dsa_batch_async(&batch_task,
                                       (const void**) &buf,
                                       1,
                                       0);
    } else {
        g_test_trap_subprocess(NULL, 0, 0);
        g_test_trap_assert_failed();
    }
}

static void test_zero_len_sync(void)
{
    if (g_test_subprocess()) {
        g_assert(!dsa_configure(path, 1));

        buffer_zero_batch_task_init(&batch_task, batch_size);

        char buf[page_size];

        buffer_is_zero_dsa_batch(&batch_task,
                                 (const void**) &buf,
                                 1,
                                 0);
    } else {
        g_test_trap_subprocess(NULL, 0, 0);
        g_test_trap_assert_failed();
    }
}

static void test_null_buf_async(void)
{
    if (g_test_subprocess()) {
        g_assert(!dsa_configure(path, 1));

        buffer_zero_batch_task_init(&batch_task, batch_size);

        buffer_is_zero_dsa_batch_async(&batch_task, NULL, 1, page_size);
    } else {
        g_test_trap_subprocess(NULL, 0, 0);
        g_test_trap_assert_failed();
    }
}

static void test_null_buf_sync(void)
{
    if (g_test_subprocess()) {
        g_assert(!dsa_configure(path, 1));

        buffer_zero_batch_task_init(&batch_task, batch_size);

        buffer_is_zero_dsa_batch(&batch_task, NULL, 1, page_size);
    } else {
        g_test_trap_subprocess(NULL, 0, 0);
        g_test_trap_assert_failed();
    }
}

static void test_batch(bool async)
{
    g_assert(!dsa_configure(path, 1));
    dsa_start();

    buffer_zero_batch_task_init(&batch_task, batch_size);

    char buf[page_size * batch_size];
    char *addrs[batch_size];
    for (int i = 0; i < batch_size; i++) {
        addrs[i] = buf + (page_size * i);
    }

    // Using whatever is on the stack is somewhat random.
    // Manually set some pages to zero and some to nonzero.
    memset(buf + 0, 0, page_size * 10);
    memset(buf + (10 * page_size), 0xff, page_size * 10);

    if (async) {
        buffer_is_zero_dsa_batch_async(&batch_task,
                                       (const void**) addrs,
                                       batch_size,
                                       page_size);   
    } else {
        buffer_is_zero_dsa_batch(&batch_task,
                                (const void**) addrs,
                                batch_size,
                                page_size);
    }

    bool is_zero;
    for (int i = 0; i < batch_size; i++) {
        is_zero = buffer_is_zero((const void*) &buf[page_size * i], page_size);
        g_assert(batch_task.results[i] == is_zero);
    }
    dsa_cleanup();
}

static void test_batch_async(void)
{
    test_batch(true);
}

static void test_batch_sync(void)
{
    test_batch(false);
}

static void test_page_fault(void)
{
    g_assert(!dsa_configure(path, 1));
    dsa_start();

    char* buf[2];
    buf[0] = (char*) mmap(NULL, page_size * batch_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    assert(buf[0] != MAP_FAILED);
    buf[1] = (char*) malloc(page_size * batch_size);
    assert(buf[1] != NULL);

    struct dsa_counters *counters;
    counters = dsa_get_counters();
    uint64_t fallback_count = counters->total_fallback_count;

    for (int j = 0; j < 2; j++) {
        buffer_zero_batch_task_init(&batch_task, batch_size);

        char *addrs[batch_size];
        for (int i = 0; i < batch_size; i++) {
            addrs[i] = buf[j] + (page_size * i);
        }

        buffer_is_zero_dsa_batch(&batch_task,
                                (const void**) addrs,
                                batch_size,
                                page_size);

        bool is_zero;
        for (int i = 0; i < batch_size; i++) {
            is_zero = buffer_is_zero((const void*) &buf[j][page_size * i], page_size);
            g_assert(batch_task.results[i] == is_zero);
        }
    }

    counters = dsa_get_counters();
    g_assert(counters->total_fallback_count > fallback_count);

    assert(!munmap(buf[0], page_size * batch_size));
    free(buf[1]);
    dsa_cleanup();
}

static void test_various_buffer_sizes(bool async)
{
    g_assert(!dsa_configure(path, 1));
    dsa_start();

    int len = 1 << 4;
    for (int count = 12; count > 0; count--, len <<= 1) {
        buffer_zero_batch_task_init(&batch_task, batch_size);

        char buf[len * batch_size];
        char *addrs[batch_size];
        for (int i = 0; i < batch_size; i++) {
            addrs[i] = buf + (len * i);
        }

        if (async) {
            buffer_is_zero_dsa_batch_async(&batch_task,
                                           (const void**) addrs,
                                           batch_size,
                                           len);
        } else {
            buffer_is_zero_dsa_batch(&batch_task,
                                    (const void**) addrs,
                                    batch_size,
                                    len);
        }

        bool is_zero;
        for (int j = 0; j < batch_size; j++) {
            is_zero = buffer_is_zero((const void*) &buf[len * j], len);
            g_assert(batch_task.results[j] == is_zero);
        }
    }
    
    dsa_cleanup();
}

static void test_various_buffer_sizes_async(void)
{
    test_various_buffer_sizes(true);
}

static void test_various_buffer_sizes_sync(void)
{
    test_various_buffer_sizes(false);
}

static void test_double_start_stop(void)
{
    g_assert(!dsa_configure(path, 1));
    // Double start
    dsa_start();
    dsa_start();
    g_assert(dsa_is_running());
    do_single_task(true);

    // Double stop
    dsa_stop();
    g_assert(!dsa_is_running());
    dsa_stop();
    g_assert(!dsa_is_running());

    // Restart
    dsa_start();
    g_assert(dsa_is_running());
    do_single_task(true);
    dsa_cleanup();
}

static void test_is_running(void)
{
    g_assert(!dsa_configure(path, 1));

    g_assert(!dsa_is_running());
    dsa_start();
    g_assert(dsa_is_running());
    dsa_stop();
    g_assert(!dsa_is_running());
    dsa_cleanup();
}

static void test_multiple_engines(void)
{
    g_assert(!dsa_configure(path, num_devices));
    dsa_start();

    struct buffer_zero_batch_task tasks[num_devices] 
        __attribute__((aligned(64)));
    char bufs[num_devices][page_size * batch_size];
    char *addrs[num_devices][batch_size];

    // This is a somewhat implementation-specific way of testing that the tasks
    // have unique engines assigned to them.
    buffer_zero_batch_task_init(&tasks[0], batch_size);
    buffer_zero_batch_task_init(&tasks[1], batch_size);
    g_assert(tasks[0].device != tasks[1].device);

    for (int i = 0; i < num_devices; i++) {
        for (int j = 0; j < batch_size; j++) {
            addrs[i][j] = bufs[i] + (page_size * j);
        }

        buffer_is_zero_dsa_batch(&tasks[i],
                                 (const void**) addrs[i],
                                 batch_size,
                                 page_size);

        bool is_zero;
        for (int j = 0; j < batch_size; j++) {
            is_zero = buffer_is_zero((const void*) &bufs[i][page_size * j], page_size);
            g_assert(tasks[i].results[j] == is_zero);
        }
    }

    dsa_cleanup();
}

static void test_configure_dsa_twice(void)
{
    g_assert(!dsa_configure(path, num_devices));
    g_assert(!dsa_configure(path, num_devices));
    dsa_start();
    do_single_task(false);
    dsa_cleanup();
}

static void test_configure_dsa_bad_path(void)
{
    const char* bad_path = "/not/a/real/path";
    g_assert(dsa_configure(&bad_path, 1));
}

static void test_cleanup_before_configure(void)
{
    dsa_cleanup();
    g_assert(!dsa_configure(path, num_devices));
}

static void test_configure_dsa_num_devices(void)
{
    g_assert(!dsa_configure(path, 0));
    g_assert(!dsa_configure(path, -1));

    g_assert(!dsa_configure(path, num_devices));
    do_single_task(false);
    dsa_cleanup();
}

static void test_cleanup_twice(void)
{
    g_assert(!dsa_configure(path, num_devices));
    dsa_cleanup();
    dsa_cleanup();

    g_assert(!dsa_configure(path, num_devices));
    dsa_start();
    do_single_task(false);
    dsa_cleanup();
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    if (getenv("QEMU_TEST_FLAKY_TESTS")) {
        g_test_add_func("/dsa/page_fault", test_page_fault);
    }

    if (num_devices > 1) {
        g_test_add_func("/dsa/multiple_engines", test_multiple_engines);
    }

    g_test_add_func("/dsa/sync/batch", test_batch_sync);
    g_test_add_func("/dsa/sync/various_buffer_sizes",
                    test_various_buffer_sizes_sync);
    g_test_add_func("/dsa/sync/null_buf", test_null_buf_sync);
    g_test_add_func("/dsa/sync/zero_len", test_zero_len_sync);
    g_test_add_func("/dsa/sync/oversized_batch", test_oversized_batch_sync);
    g_test_add_func("/dsa/sync/zero_count", test_zero_count_sync);
    g_test_add_func("/dsa/sync/single_zero", test_single_zero_sync);
    g_test_add_func("/dsa/sync/single_nonzero", test_single_nonzero_sync);
    g_test_add_func("/dsa/sync/null_task", test_null_task_sync);

    g_test_add_func("/dsa/async/batch", test_batch_async);
    g_test_add_func("/dsa/async/various_buffer_sizes",
                    test_various_buffer_sizes_async);
    g_test_add_func("/dsa/async/null_buf", test_null_buf_async);
    g_test_add_func("/dsa/async/zero_len", test_zero_len_async);
    g_test_add_func("/dsa/async/oversized_batch", test_oversized_batch_async);
    g_test_add_func("/dsa/async/zero_count", test_zero_count_async);
    g_test_add_func("/dsa/async/single_zero", test_single_zero_async);
    g_test_add_func("/dsa/async/single_nonzero", test_single_nonzero_async);
    g_test_add_func("/dsa/async/null_task", test_null_task_async);

    g_test_add_func("/dsa/double_start_stop", test_double_start_stop);
    g_test_add_func("/dsa/is_running", test_is_running);

    g_test_add_func("/dsa/configure_dsa_twice", test_configure_dsa_twice);
    g_test_add_func("/dsa/configure_dsa_bad_path", test_configure_dsa_bad_path);
    g_test_add_func("/dsa/cleanup_before_configure",
                    test_cleanup_before_configure);
    g_test_add_func("/dsa/configure_dsa_num_devices",
                    test_configure_dsa_num_devices);
    g_test_add_func("/dsa/cleanup_twice", test_cleanup_twice);

    return g_test_run();
}
