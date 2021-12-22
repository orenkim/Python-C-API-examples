#ifndef _CPYTOOLS_ROLLING_HPP
#define _CPYTOOLS_ROLLING_HPP

#include <thread>
#include <vector>


using rolling_function_type = void(*)(const double *, long, double *, long, long);


static void apply_rolling_function_column_range
(rolling_function_type rolling_function, const double *in_ptr, long window, double *out_ptr,
long length, long stride, long column_start, long column_stop)
{
    for (long i = column_start; i < column_stop; ++i) {
        rolling_function(in_ptr + i, window, out_ptr + i, length, stride);
    }
}


static void apply_rolling_function
(rolling_function_type rolling_function, const double *in_ptr, long window, double *out_ptr,
long num_rows, long num_columns)
{
    apply_rolling_function_column_range(rolling_function, in_ptr, window, out_ptr, num_rows, num_columns, 0, num_columns);
}


static void apply_rolling_function_multi_thread
(rolling_function_type rolling_function, const double *in_ptr, long window, double *out_ptr,
long num_rows, long num_columns)
{
    std::vector<std::thread> thread_vec;
    long start, stop, i;
    long num_threads = (std::thread::hardware_concurrency() + 1) / 2;

    // Columns are divided into (num_thread) pieces, each of which is assigned to a thread.

    long per_thread = num_columns / num_threads;
    long rem = num_columns % num_threads;

    for (i = 0; i < num_threads; ++i) {
        if (i < rem) {
            start = (per_thread + 1) * i;
            stop = (per_thread + 1) * (i + 1);
        }
        else {
            start = per_thread * i + rem;
            stop = per_thread * (i + 1) + rem;
        }

        if (start < stop) { // could be false if num_columns < num_threads
            // Each thread executes "apply_rolling_function_column_range(rolling_function, in_ptr, window, out_ptr, num_rows, num_columns, start, stop)"
            thread_vec.push_back(std::thread(apply_rolling_function_column_range, rolling_function,
                                             in_ptr, window, out_ptr, num_rows, num_columns, start, stop));
        }
    }

    for (auto &thread : thread_vec) {
        thread.join();
    }
}

#endif
