#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>

static PyObject *walk_impl(PyObject *self, PyObject *args) {
    PyObject *rowptr_obj, *col_obj, *start_obj;
    int64_t walk_length;
    int allow_backtrack;

    if (!PyArg_ParseTuple(args, "OOOli", &rowptr_obj, &col_obj, &start_obj, &walk_length, &allow_backtrack))
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
        int64_t curr = starts[i];
        int64_t prev = -1;
        data[i * walk_length] = curr;
        for (int64_t k = 1; k < walk_length; k++) {
            int64_t rs = rowptr[curr];
            int64_t re = rowptr[curr + 1];
            int64_t degree = re - rs;
            int64_t nxt;

            if (degree == 0) {
                data[i * walk_length + k] = curr;
                continue;
            } else if (!allow_backtrack && prev >= 0 && degree > 1) {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                int64_t idx = (int64_t)(state >> 33) % (degree - 1);
                int64_t count = 0;
                nxt = curr;
                for (int64_t j = rs; j < re; j++) {
                    if (col[j] == prev) continue;
                    if (count == idx) { nxt = col[j]; break; }
                    count++;
                }
            } else {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                nxt = col[rs + (int64_t)(state >> 33) % degree];
            }

            prev = curr;
            curr = nxt;
            data[i * walk_length + k] = curr;
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
