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
static void do_single_task(void)
{
    buffer_zero_batch_task_init(&batch_task, batch_size);
    char buf[page_size];
    char* ptr = buf;

    buffer_is_zero_dsa_batch(&batch_task,
                             (const void**) &ptr,
                             1,
                             page_size);
    g_assert(batch_task.results[0] == buffer_is_zero(buf, page_size));
}

static void test_single_zero(void)
{
    g_assert(!dsa_configure(path, 1));

    buffer_zero_batch_task_init(&batch_task, batch_size);

    char buf[page_size];
    char* ptr = buf;

    memset(buf, 0x0, page_size);
    buffer_is_zero_dsa_batch(&batch_task,
                             (const void**) &ptr,
                             1,
                             page_size);
    g_assert(batch_task.results[0]);
    
    dsa_cleanup();
}

static void test_single_nonzero(void)
{
    g_assert(!dsa_configure(path, 1));

    buffer_zero_batch_task_init(&batch_task, batch_size);

    char buf[page_size];
    char* ptr = buf;

    memset(buf, 0x1, page_size);
    buffer_is_zero_dsa_batch(&batch_task,
                             (const void**) &ptr,
                             1,
                             page_size);
    g_assert(!batch_task.results[0]);
    
    dsa_cleanup(); 
}

// count == 0 should return quickly without calling into DSA.
static void test_zero_count(void)
{
    char buf[page_size];
    buffer_is_zero_dsa_batch(&batch_task,
                             (const void **) &buf,
                             0,
                             page_size);
}

static void test_null_task(void)
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

static void test_oversized_batch(void)
{
    g_assert(!dsa_configure(path, 1));

    buffer_zero_batch_task_init(&batch_task, batch_size);

    int oversized_batch_size = batch_size + 1;
    char buf[page_size * oversized_batch_size];
    char *addrs[batch_size];
    for (int i = 0; i < oversized_batch_size; i++) {
        addrs[i] = buf + (page_size * i);
    }

    int ret;
    ret = buffer_is_zero_dsa_batch(&batch_task,
                                   (const void**) addrs,
                                   oversized_batch_size,
                                   page_size);
    g_assert(ret != 0);

    dsa_cleanup();
}

static void test_zero_len(void)
{
    if (g_test_subprocess()) {
        g_assert(!dsa_configure(path, 1));

        buffer_zero_batch_task_init(&batch_task, batch_size);

        char buf[page_size];

        buffer_is_zero_dsa_batch(&batch_task, (const void**) &buf, 1, 0);
    } else {
        g_test_trap_subprocess(NULL, 0, 0);
        g_test_trap_assert_failed();
    }
}

static void test_null_buf(void)
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

static void test_batch(void)
{
    g_assert(!dsa_configure(path, 1));

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

    buffer_is_zero_dsa_batch(&batch_task,
                             (const void**) addrs,
                             batch_size,
                             page_size);

    bool is_zero;
    for (int i = 0; i < batch_size; i++) {
        is_zero = buffer_is_zero((const void*) &buf[page_size * i], page_size);
        g_assert(batch_task.results[i] == is_zero);
    }
    dsa_cleanup();
}

static void test_page_fault(void)
{
    g_assert(!dsa_configure(path, 1));

    char* buf[2];
    buf[0] = (char*) mmap(NULL, page_size * batch_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    assert(buf[0] != MAP_FAILED);
    buf[1] = (char*) malloc(page_size * batch_size);
    assert(buf[1] != NULL);

    uint64_t fallback_count = dsa_counters.total_fallback_count;

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

    g_assert(dsa_counters.total_fallback_count > fallback_count);

    assert(!munmap(buf[0], page_size * batch_size));
    free(buf[1]);
    dsa_cleanup();
}

static void test_various_buffer_sizes(void)
{
    g_assert(!dsa_configure(path, 1));

    int len = 1 << 4;
    for (int count = 12; count > 0; count--, len <<= 1) {
        buffer_zero_batch_task_init(&batch_task, batch_size);

        char buf[len * batch_size];
        char *addrs[batch_size];
        for (int i = 0; i < batch_size; i++) {
            addrs[i] = buf + (len * i);
        }

        buffer_is_zero_dsa_batch(&batch_task,
                                (const void**) addrs,
                                batch_size,
                                len);

        bool is_zero;
        for (int j = 0; j < batch_size; j++) {
            is_zero = buffer_is_zero((const void*) &buf[len * j], len);
            g_assert(batch_task.results[j] == is_zero);
        }
    }
    
    dsa_cleanup();
}

static void test_multiple_engines(void)
{
    g_assert(!dsa_configure(path, num_devices));

    struct buffer_zero_batch_task tasks[num_devices] 
        __attribute__((aligned(64)));
    char bufs[num_devices][page_size * batch_size];
    char *addrs[num_devices][batch_size];

    // This is a somewhat implementation-specific way of testing that the tasks
    // have unique engines assigned to them.
    g_assert(tasks[0].device != tasks[1].device);

    for (int i = 0; i < num_devices; i++) {
        buffer_zero_batch_task_init(&tasks[i], batch_size);

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
    do_single_task();
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
    g_assert(dsa_configure(path, 0));
    g_assert(dsa_configure(path, -1));

    g_assert(!dsa_configure(path, num_devices));
    do_single_task();
    dsa_cleanup();
}

static void test_cleanup_twice(void)
{
    g_assert(!dsa_configure(path, num_devices));
    dsa_cleanup();
    dsa_cleanup();

    g_assert(!dsa_configure(path, num_devices));
    do_single_task();
    dsa_cleanup();
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/dsa/batch", test_batch);
    g_test_add_func("/dsa/various_buffer_sizes", test_various_buffer_sizes);
    if (getenv("QEMU_TEST_FLAKY_TESTS")) {
        g_test_add_func("/dsa/page_fault", test_page_fault);
    }
    g_test_add_func("/dsa/null_buf", test_null_buf);
    g_test_add_func("/dsa/zero_len", test_zero_len);
    g_test_add_func("/dsa/oversized_batch", test_oversized_batch);
    g_test_add_func("/dsa/zero_count", test_zero_count);
    g_test_add_func("/dsa/single_zero", test_single_zero);
    g_test_add_func("/dsa/single_nonzero", test_single_nonzero);
    g_test_add_func("/dsa/null_task", test_null_task);
    if (num_devices > 1) {
        g_test_add_func("/dsa/multiple_engines", test_multiple_engines);
    }

    g_test_add_func("/dsa/configure_dsa_twice", test_configure_dsa_twice);
    g_test_add_func("/dsa/configure_dsa_bad_path", test_configure_dsa_bad_path);
    g_test_add_func("/dsa/cleanup_before_configure",
                    test_cleanup_before_configure);
    g_test_add_func("/dsa/configure_dsa_num_devices",
                    test_configure_dsa_num_devices);
    g_test_add_func("/dsa/cleanup_twice", test_cleanup_twice);

    return g_test_run();
}
