/*
 * prox_cross_entropy.c
 *
 *  Created on: Dec 10, 2013
 *      Author: vikas
 */

#include "Python.h"
#include "arrayobject.h"
#include "libproxoperators.h"
#include <math.h>

static PyMethodDef _libproxoperatorsMethods[] = {
		{"crossentropy_prox", crossentropy_prox, METH_VARARGS},
		{"crossentropy_obj", crossentropy_obj, METH_VARARGS},
		{"hinge_prox", hinge_prox, METH_VARARGS},
		{"hinge_obj",  hinge_obj, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

void init_libproxoperators()  {
	(void) Py_InitModule("_libproxoperators", _libproxoperatorsMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject *crossentropy_prox(PyObject *self, PyObject *args)
{
	PyArrayObject *x, *v, *y;
	double lambda, epsilon;
	int MAXITER, m, n, flag, i, DISPLAY;  // ncomps=n*m=total number of matrix components in mat1

	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "OOdOidi",
		&y, &v, &lambda, &x, &MAXITER, &epsilon,&DISPLAY))  return NULL;
	if (NULL == v)  return NULL;

	/* Check that object input is 'double' type and a matrix
	   Not needed if python wrapper function checks before call to this routine */
	if (v->descr->type_num != NPY_DOUBLE || v->nd != 2)  {
			PyErr_SetString(PyExc_ValueError,
				"In not_doublematrix: array must be of type Float and 2 dimensional (m x n).");
			return NULL;  }

//	printf("y: %d, %d", y->descr->type_num, y->nd);
//	if (y->descr->type_num != NPY_DOUBLE || y->nd != 2)  {
	//			PyErr_SetString(PyExc_ValueError,
	//				"In not_doublematrix: array must be of type Float and 2 dimensional (m x 1).");
	//			return NULL;  }

	/* Get the dimensions of the input */

	PyObject *v_array = PyArray_FROM_OTF((PyObject *) v, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject *y_array = PyArray_FROM_OTF((PyObject *) y, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject *x_array = PyArray_FROM_OTF((PyObject *) x, NPY_DOUBLE, NPY_INOUT_ARRAY);

	if (x_array == NULL || v_array == NULL) {
	         Py_XDECREF(x_array);
	         Py_XDECREF(y_array);
	         Py_XDECREF(v_array);
	         return NULL;
	     }

 //   printf("Number of dimensions=%d\n", PyArray_NDIM(v_array));

    m = (int) PyArray_DIM(v_array,0);
	n = (int) PyArray_DIM(v_array,1);

	double *cx    = (double*)PyArray_DATA(x_array);
	double *cy = (double*) PyArray_DATA(y_array);
	double *cv    = (double*)PyArray_DATA(v_array);

//	printf("m=%d, n=%d,lambda=%f,MAXITER=%d,epsilon=%f\n", m, n, lambda, MAXITER, epsilon);
//	for(i=0;i<m;i++)
//		printf("%f ", cy[i]);
//	printf("\n");

	for(i=0;i<m;i++)
		flag = logexp((int) cy[i], cv + i*n, n, lambda, cx + i*n, MAXITER, epsilon, DISPLAY);

	Py_DECREF(x_array);
	Py_DECREF(v_array);
	Py_DECREF(y_array);

	// printf("returned %d", flag);

	PyObject *ret = Py_BuildValue("i", flag);
	return ret;
}


static PyObject *hinge_prox(PyObject *self, PyObject *args)
{
	PyArrayObject *x, *v, *y;
	double lambda, yv, yy, *cvv, *cxx;
	int  m, n, i, j,label;  // ncomps=n*m=total number of matrix components in mat1

	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "OOdO",
		&y, &v, &lambda, &x))  return NULL;
	if (NULL == v)  return NULL;

	/* Check that object input is 'double' type and a matrix
	   Not needed if python wrapper function checks before call to this routine */
	if (v->descr->type_num != NPY_DOUBLE || v->nd != 2)  {
			PyErr_SetString(PyExc_ValueError,
				"In not_doublematrix: array must be of type Float and 2 dimensional (m x n).");
			return NULL;  }

//	printf("y: %d, %d", y->descr->type_num, y->nd);
//	if (y->descr->type_num != NPY_DOUBLE || y->nd != 2)  {
	//			PyErr_SetString(PyExc_ValueError,
	//				"In not_doublematrix: array must be of type Float and 2 dimensional (m x 1).");
	//			return NULL;  }

	/* Get the dimensions of the input */

	PyObject *v_array = PyArray_FROM_OTF((PyObject *) v, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject *y_array = PyArray_FROM_OTF((PyObject *) y, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject *x_array = PyArray_FROM_OTF((PyObject *) x, NPY_DOUBLE, NPY_INOUT_ARRAY);

	if (x_array == NULL || v_array == NULL) {
	         Py_XDECREF(x_array);
	         Py_XDECREF(y_array);
	         Py_XDECREF(v_array);
	         return NULL;
	     }

 //   printf("Number of dimensions=%d\n", PyArray_NDIM(v_array));

    m = (int) PyArray_DIM(v_array,0);
	n = (int) PyArray_DIM(v_array,1);

	double *cx    = (double*)PyArray_DATA(x_array);
	double *cy = (double*) PyArray_DATA(y_array);
	double *cv    = (double*)PyArray_DATA(v_array);

//	printf("m=%d, n=%d,lambda=%f,MAXITER=%d,epsilon=%f\n", m, n, lambda, MAXITER, epsilon);
//	for(i=0;i<m;i++)
//		printf("%f ", cy[i]);
//	printf("\n");
	//flag = logexp((int) cy[i], cv + i*n, n, lambda, cx + i*n);
	if(n==1) { // We assume cy has +1 or -1 entries for n=1 outputs
		for(i=0;i<m;i++) {
			yv = cy[i]*cv[i];
			if (yv>1.0)
				cx[i] = cv[i];
			else {
				if(yv<1.0-lambda)
					cx[i] = cv[i] + lambda*cy[i];
				else
					cx[i] = cy[i];
			}
		}
	}

	if (n>1) {
		for(i=0;i<m;i++) {
			label = (int) cy[i];
			cvv = cv + i*n;
			cxx = cx + i*n;
			for(j=0;j<n;j++) {
				yv = cvv[j];
				yy = +1.0;
				if(!(j==label)) {
					yv = -yv;
					yy = -1.0;
				}
				if (yv>1.0)
								cxx[j] = cvv[j];
							else {
								if(yv<1.0-lambda)
									cxx[j] = cvv[j] + lambda*yy;
								else
									cxx[j] = yy;
							}
			}
		}
	}

	Py_DECREF(x_array);
	Py_DECREF(v_array);
	Py_DECREF(y_array);

	// printf("returned %d", flag);

	PyObject *ret = Py_BuildValue("i", 1.0);
	return ret;
}


int logexp(int index, double* v, int n, double lambda, double* x, int MAXITER, double epsilon, int DISPLAY) {
	/* solution to - log exp(x(i))/sum(exp(x(j))) + lambda/2 ||x - v||_2^2 */
	/* n is length of v and x */
	/* writes over x */
	double alpha = 0.1;
	double beta = 0.5;
	int iter, i;
	double t, logsum, p, pu, pptil, decrement;
	double *u = (double *) malloc(n*sizeof(double));
	double *z = (double *) malloc(n*sizeof(double));
	double *grad = (double *) malloc(n*sizeof(double));
	double newobj=0.0, obj=0.0;
	obj = objective(index, x, v, n, lambda);

	for(iter=0;iter<MAXITER;iter++) {
		logsum = logsumexp(x,n);
		if(DISPLAY)
			printf("iter=%d, obj=%f\n", iter, obj);
		pu = 0.0;
		pptil = 0.0;
		for(i=0;i<n;i++) {
			p = exp(x[i] - logsum);
			grad[i] = p + lambda*(x[i] - v[i]);
			if(i==index)
				grad[i] += -1.0;
			u[i] = grad[i]/(p+lambda);
			pu += p*u[i];
			z[i] = p/(p+lambda);
			pptil += z[i]*p;
		}
		decrement = 0.0;
		for(i=0;i<n;i++) {
			u[i] -= (pu/pptil)*z[i];
			decrement += grad[i]*u[i];
		}
		if (decrement < 2*epsilon) {
			free(u);
			free(z);
			free(grad);
			return 0;
		}
		t = 1.0;
		while(1) {
			for(i=0;i<n;i++)
				z[i] = x[i] - t*u[i];
			newobj = objective(index, z, v, n, lambda);
			if (newobj <= obj + alpha*t*decrement)
				break;
			t = beta*t;
		}
		for(i=0;i<n;i++)
			x[i] = z[i];
			obj = newobj;
	}
	free(u);
	free(z);
	free(grad);
	return 1;
}


double objective(int index, double* x, double* v, int n, double lambda) {
	double nrmsqr = normsquare(x,v,n);
	double obj = -x[index] + logsumexp(x, n) + 0.5*lambda*nrmsqr;
	return obj;
}

double normsquare(double* x, double* y, int n){
	double nrm = 0.0;
	int i;
	for(i=0;i<n;i++)
		nrm+= pow(x[i] - y[i], 2);
	return nrm;
}

double logsumexp(double* x, int n) {
	int i;
	double max=x[0];
	double f = 0.0;
	for(i=0;i<n;i++) {
		if (x[i]>max) {
			max = x[i];
		}
	}
	for(i=0;i<n;i++)
		f +=  exp(x[i] - max);
	f = max + log(f);

	return f;
}


static PyObject *crossentropy_obj(PyObject *self, PyObject *args)
{
	PyArrayObject *x, *y;
	int m,n, i;
	double obj;
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "OO",
		&y, &x))  return NULL;
	if (NULL == x)  return NULL;

	/* Check that object input is 'double' type and a matrix
	   Not needed if python wrapper function checks before call to this routine */
	if (x->descr->type_num != NPY_DOUBLE || x->nd != 2)  {
			PyErr_SetString(PyExc_ValueError,
				"In not_doublematrix: array must be of type Float and 2 dimensional (m x n).");
			return NULL;  }

	if (y->descr->type_num != NPY_DOUBLE || y->nd != 2)  {
				PyErr_SetString(PyExc_ValueError,
					"In not_doublematrix: array must be of type Float and 2 dimensional (m x n).");
				return NULL;  }

//	printf("y: %d, %d", y->descr->type_num, y->nd);
//	if (y->descr->type_num != NPY_DOUBLE || y->nd != 2)  {
	//			PyErr_SetString(PyExc_ValueError,
	//				"In not_doublematrix: array must be of type Float and 2 dimensional (m x 1).");
	//			return NULL;  }

	/* Get the dimensions of the input */

	PyObject *y_array = PyArray_FROM_OTF((PyObject *) y, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject *x_array = PyArray_FROM_OTF((PyObject *) x, NPY_DOUBLE, NPY_IN_ARRAY);

	if (x_array == NULL || y_array == NULL) {
	         Py_XDECREF(x_array);
	         Py_XDECREF(y_array);
	         return NULL;
	     }

 //   printf("Number of dimensions=%d\n", PyArray_NDIM(v_array));

    m = (int) PyArray_DIM(x_array,0);
	n = (int) PyArray_DIM(x_array,1);

	double *cx    = (double*)PyArray_DATA(x_array);
	double *cy = (double*) PyArray_DATA(y_array);

	obj = 0.0;
	for(i=0;i<m;i++)
		obj += -*(cx + i*n + (int) cy[i]) + logsumexp(cx + i*n, n);


	Py_DECREF(x_array);
	Py_DECREF(y_array);

	// printf("returned %d", flag);

	PyObject *ret = Py_BuildValue("d", obj);
	return ret;
}

static PyObject *hinge_obj(PyObject *self, PyObject *args)
{
	PyArrayObject *x, *y;
	int m,n, i,j, label;
	double obj, yx, yy, *cxx;
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "OO",
		&y, &x))  return NULL;
	if (NULL == x)  return NULL;

	/* Check that object input is 'double' type and a matrix
	   Not needed if python wrapper function checks before call to this routine */
	if (x->descr->type_num != NPY_DOUBLE || x->nd != 2)  {
			PyErr_SetString(PyExc_ValueError,
				"In not_doublematrix: array must be of type Float and 2 dimensional (m x n).");
			return NULL;  }

	if (y->descr->type_num != NPY_DOUBLE || y->nd != 2)  {
				PyErr_SetString(PyExc_ValueError,
					"In not_doublematrix: array must be of type Float and 2 dimensional (m x n).");
				return NULL;  }

//	printf("y: %d, %d", y->descr->type_num, y->nd);
//	if (y->descr->type_num != NPY_DOUBLE || y->nd != 2)  {
	//			PyErr_SetString(PyExc_ValueError,
	//				"In not_doublematrix: array must be of type Float and 2 dimensional (m x 1).");
	//			return NULL;  }

	/* Get the dimensions of the input */

	PyObject *y_array = PyArray_FROM_OTF((PyObject *) y, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject *x_array = PyArray_FROM_OTF((PyObject *) x, NPY_DOUBLE, NPY_IN_ARRAY);

	if (x_array == NULL || y_array == NULL) {
	         Py_XDECREF(x_array);
	         Py_XDECREF(y_array);
	         return NULL;
	     }

 //   printf("Number of dimensions=%d\n", PyArray_NDIM(v_array));

    m = (int) PyArray_DIM(x_array,0);
	n = (int) PyArray_DIM(x_array,1);

	double *cx    = (double*)PyArray_DATA(x_array);
	double *cy = (double*) PyArray_DATA(y_array);

	obj = 0.0;

	if(n==1) {
		for(i=0;i<m;i++) {
			yx = cx[i]*cy[i];
			if(yx<1.0)
				obj += (1.0 - yx);
		}
	}


	if(n>1) {
			for(i=0;i<m;i++) {
				label = (int) cy[i];
				cxx = cx + i*n;
				for(j=0;j<n;j++) {
					yy = -1.0;
					if (j==label)
						yy = +1.0;
					yx = cxx[j]*yy;
					if(yx<1.0)
						obj += (1.0 - yx);
				}
			}
		}


	Py_DECREF(x_array);
	Py_DECREF(y_array);

	// printf("returned %d", flag);

	PyObject *ret = Py_BuildValue("d", obj);
	return ret;
}
