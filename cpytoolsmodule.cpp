#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <thread>
#include <vector>


static void rolling_sum_single_column(const double *in_ptr, long window, double *out_ptr, long length, long stride) {
    long i;
    double sum = 0.0;
    long cnt = 0, idx;
    double val, val_adj, new_sum, delta = 0.0;

    for (i = 0; i < window; ++i) {
        idx = i * stride;
        val = in_ptr[idx];

        if (std::isfinite(val)) {
            // Adds new value
            // Kahan summation algorithm
            val_adj = val - delta;
            new_sum = sum + val_adj;
            delta = (new_sum - sum) - val_adj;
            sum = new_sum;
            ++cnt;
        }

        if (cnt >= 1) {
            out_ptr[idx] = sum;
        }
        else {
            out_ptr[idx] = NAN;  // output = NAN when all values in the window are NAN
        }
    }

    for (i = window; i < length; ++i) {
        idx = i * stride;
        val = in_ptr[idx - window * stride];

        if (std::isfinite(val)) {
            // Subtracts old value
            // Kahan summation algorithm
            val_adj = val + delta;
            new_sum = sum - val_adj;
            delta = (new_sum - sum) + val_adj;
            sum = new_sum;
            --cnt;
        }

        val = in_ptr[idx];

        if (std::isfinite(val)) {
            // Adds new value
            // Kahan summation algorithm
            val_adj = val - delta;
            new_sum = sum + val_adj;
            delta = (new_sum - sum) - val_adj;
            sum = new_sum;
            ++cnt;
        }

        if (cnt >= 1) {
            out_ptr[idx] = sum;
        }
        else {
            out_ptr[idx] = NAN; // output = NAN when all values in the window are NAN
        }
    }
}


static void rolling_sum_column_range(const double *in_ptr, long window, double *out_ptr, long length, long stride,
                                     long start, long stop) {
    // Applies "rolling_sum_single_column" on the range [start: stop]
    for (long i = start; i < stop; ++i) {
        rolling_sum_single_column(in_ptr + i, window, out_ptr + i, length, stride);
    }
}


static void rolling_sum_2d_single_thread(const double *in_ptr, long window, double *out_ptr,
                                         long num_rows, long num_cols) {
    rolling_sum_column_range(in_ptr, window, out_ptr, num_rows, num_cols, 0, num_cols);
}


static void rolling_sum_2d_multi_thread(const double *in_ptr, long window, double *out_ptr,
                                        long num_rows, long num_cols) {
    std::vector<std::thread> thread_vec;
    long start, stop, i;
    long num_threads = (std::thread::hardware_concurrency() + 1) / 2;

    // Columns are divided into (num_thread) pieces, each of which is assigned to a thread.

    long per_thread = num_cols / num_threads;
    long rem = num_cols % num_threads;

    for (i = 0; i < num_threads; ++i) {
        if (i < rem) {
            start = (per_thread + 1) * i;
            stop = (per_thread + 1) * (i + 1);
        }
        else {
            start = per_thread * i + rem;
            stop = per_thread * (i + 1) + rem;
        }

        // Each thread executes "rolling_sum_column_range(in_ptr, window, out_ptr, num_rows, num_cols, start, stop)"
        thread_vec.push_back(std::thread(rolling_sum_column_range, in_ptr, window, out_ptr, num_rows, num_cols, start, stop));
    }

    for (auto &&thread : thread_vec) {
        thread.join();
    }
}


static PyObject *calculate_rolling_stats(PyObject *args, void(*calculator_function)(const double *, long, double *, long, long)) {
    PyObject *in_arg = nullptr;
    long window;
    if (!PyArg_ParseTuple(args, "Ol", &in_arg, &window)) {
        return nullptr;
    }

    PyObject *in_arr = PyArray_FROM_OTF(in_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!in_arr) return nullptr;
    PyArrayObject *in_arr_obj = reinterpret_cast<PyArrayObject *>(in_arr);

    npy_intp *dims = PyArray_DIMS(in_arr_obj);
    int nd = PyArray_NDIM(in_arr_obj);

    if (nd != 2) {
        PyErr_SetString(PyExc_ValueError, "1st argument must be a numpy 2d array or an object that can be converted to such.");
        Py_DECREF(in_arr);
        return nullptr;
    }

    if (window <= 0) {
        PyErr_SetString(PyExc_ValueError, "2nd argument must be a positive integer.");
        Py_DECREF(in_arr);
        return nullptr;
    }

    PyObject *out_arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    auto in_ptr = reinterpret_cast<const double *>(PyArray_DATA(in_arr_obj));
    auto out_ptr = reinterpret_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(out_arr)));
    long num_rows = dims[0], num_cols = dims[1];

    calculator_function(in_ptr, window, out_ptr, num_rows, num_cols);

    Py_DECREF(in_arr);

    return out_arr;
}


static PyObject *rolling_sum(PyObject *self, PyObject *args) {
    return calculate_rolling_stats(args, rolling_sum_2d_single_thread);
}


static PyObject *rolling_sum2(PyObject *self, PyObject *args) {
    return calculate_rolling_stats(args, rolling_sum_2d_multi_thread);
}


static PyMethodDef CPyToolsMethods[] = {
    {"rolling_sum",  rolling_sum, METH_VARARGS, "calculates rolling sum"},
    {"rolling_sum2",  rolling_sum2, METH_VARARGS, "calculates rolling sum using multiple threads"},
    {nullptr, nullptr, 0, nullptr}
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cpytools",
    nullptr,
    -1,
    CPyToolsMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};


PyMODINIT_FUNC PyInit_cpytools(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return nullptr;
    }

    import_array();
    return m;
}
