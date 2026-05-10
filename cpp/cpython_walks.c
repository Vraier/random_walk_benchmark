#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>

static PyObject *walk_impl(PyObject *self, PyObject *args) {
    PyObject *rowptr_obj, *col_obj, *start_obj;
    int64_t walk_length;

    if (!PyArg_ParseTuple(args, "OOOl", &rowptr_obj, &col_obj, &start_obj, &walk_length))
        return NULL;

    Py_buffer rowptr_buf, col_buf, start_buf;
    if (PyObject_GetBuffer(rowptr_obj, &rowptr_buf, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(col_obj,    &col_buf,    PyBUF_SIMPLE) < 0) { PyBuffer_Release(&rowptr_buf); return NULL; }
    if (PyObject_GetBuffer(start_obj,  &start_buf,  PyBUF_SIMPLE) < 0) { PyBuffer_Release(&rowptr_buf); PyBuffer_Release(&col_buf); return NULL; }

    int64_t *rowptr = (int64_t *)rowptr_buf.buf;
    int64_t *col    = (int64_t *)col_buf.buf;
    int64_t *starts = (int64_t *)start_buf.buf;
    int64_t  n_walks = (int64_t)(start_buf.len / sizeof(int64_t));

    npy_intp dims[2] = {(npy_intp)n_walks, (npy_intp)walk_length};
    PyObject *result = PyArray_SimpleNew(2, dims, NPY_INT64);
    if (!result) goto cleanup;
    int64_t *data = (int64_t *)PyArray_DATA((PyArrayObject *)result);

    /* simple LCG for thread-safe, dependency-free RNG */
    uint64_t state = 42;
    for (int64_t i = 0; i < n_walks; i++) {
        int64_t node = starts[i];
        data[i * walk_length] = node;
        for (int64_t k = 1; k < walk_length; k++) {
            int64_t rs = rowptr[node];
            int64_t re = rowptr[node + 1];
            if (rs == re) {
                data[i * walk_length + k] = node;
            } else {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                node = col[rs + (int64_t)(state >> 33) % (re - rs)];
                data[i * walk_length + k] = node;
            }
        }
    }

cleanup:
    PyBuffer_Release(&rowptr_buf);
    PyBuffer_Release(&col_buf);
    PyBuffer_Release(&start_buf);
    return result;
}

static PyMethodDef CpythonWalksMethods[] = {
    {"walk_impl", walk_impl, METH_VARARGS, "Sequential random walk (CPython C extension)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef cpython_walks_module = {
    PyModuleDef_HEAD_INIT, "cpython_walks", NULL, -1, CpythonWalksMethods,
};

PyMODINIT_FUNC PyInit_cpython_walks(void) {
    import_array();
    return PyModule_Create(&cpython_walks_module);
}
