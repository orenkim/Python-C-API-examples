#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include "rolling.hpp"


using rolling_function_type = void(*)(const double *, long, double *, long, long);
using rolling_applier_type = void(*)(rolling_function_type, const double *, long, double *, long, long);


static void rolling_sum(const double *in_ptr, long window, double *out_ptr, long length, long stride) {
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


static PyObject *calculate_rolling_stats
(rolling_function_type rolling_function, rolling_applier_type rolling_applier, PyObject *args)
{
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
    long num_rows = dims[0];
    long num_columns = dims[1];

    rolling_applier(rolling_function, in_ptr, window, out_ptr, num_rows, num_columns);

    Py_DECREF(in_arr);

    return out_arr;
}


static PyObject *cpytools_rolling_sum(PyObject *self, PyObject *args) {
    return calculate_rolling_stats(rolling_sum, apply_rolling_function, args);
}


static PyObject *cpytools_rolling_sum2(PyObject *self, PyObject *args) {
    return calculate_rolling_stats(rolling_sum, apply_rolling_function_multi_thread, args);
}


static PyMethodDef CPyToolsMethods[] = {
    {"rolling_sum",  cpytools_rolling_sum, METH_VARARGS, "calculates rolling sum"},
    {"rolling_sum2",  cpytools_rolling_sum2, METH_VARARGS, "calculates rolling sum using multiple threads"},
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
