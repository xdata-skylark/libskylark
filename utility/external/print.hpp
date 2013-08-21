#ifndef PRINT_HPP
#define PRINT_HPP

#include <cstdio>

#include "../../config.h"

#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#include "FullyDistMultiVec.hpp"
#endif

namespace skylark { namespace utility {

template <typename DataType> 
struct print_t { };

#if SKYLARK_HAVE_ELEMENTAL

template <typename ValueType>
struct print_t <elem::Matrix<ValueType> > {

  typedef int index_t;
  typedef ValueType value_t;
  typedef elem::Matrix<ValueType> matrix_t;

  static void apply(const matrix_t& X,
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;

    if (am_i_printing) {
      printf ("Dump of %s\n", msg);
      for (index_t i=0; i<X.Height(); ++i) {
        for (index_t j=0; j<X.Width(); ++j) {
          printf ("%lf ", X.Get(i,j));
        }
        printf ("\n");
      }
    }
  }

  static void apply(const matrix_t& X,
                    const matrix_t& Y,
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;

    if (am_i_printing) {
      printf ("Dump of %s\n", msg);
      for (index_t i=0; i<X.Height(); ++i) {
        for (index_t j=0; j<X.Width(); ++j) {
          printf ("(%lf -- %lf) ", X.Get(i,j), Y.Get(i,j));
        }
        printf ("\n");
      }
    }
  }
};

template <typename ValueType,
          elem::Distribution CD,
          elem::Distribution RD>
struct print_t <elem::DistMatrix<ValueType, CD, RD> > {

  typedef int index_t;
  typedef ValueType value_t;
  typedef elem::DistMatrix<ValueType, CD, RD> mpi_matrix_t;
  typedef elem::Matrix<ValueType> matrix_t;

  static void apply(const mpi_matrix_t& X,
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;

    elem::AxpyInterface<value_t> interface;
    interface.Attach (elem::GLOBAL_TO_LOCAL, X);
    if (am_i_printing) {
      matrix_t local_X (X.Height(), X.Width()); 
      elem::MakeZeros (local_X);
      interface.Axpy (1.0, local_X, 0, 0);

        
      printf ("Dump of %s\n", msg);
      for (index_t i=0; i<local_X.Height(); ++i) {
        for (index_t j=0; j<local_X.Width(); ++j) {
          printf ("%lf ", local_X.Get(i,j));
        }
        printf ("\n");
      }
    }
    interface.Detach();
  }

  static void apply(const mpi_matrix_t& X,
                    const mpi_matrix_t& Y,
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;

    elem::AxpyInterface<value_t> interface_1;
    interface_1.Attach (elem::GLOBAL_TO_LOCAL, X);
    elem::AxpyInterface<value_t> interface_2;
    interface_2.Attach (elem::GLOBAL_TO_LOCAL, Y);
    if (am_i_printing) {
      matrix_t local_X (X.Height(), X.Width()); 
      matrix_t local_Y (Y.Height(), Y.Width()); 
      elem::MakeZeros (local_X);
      elem::MakeZeros (local_Y);
      interface_1.Axpy (1.0, local_X, 0, 0);
      interface_2.Axpy (1.0, local_Y, 0, 0);
        
      printf ("Dump of %s\n", msg);
      for (index_t i=0; i<local_X.Height(); ++i) {
        for (index_t j=0; j<local_X.Width(); ++j) {
          printf ("(%lf -- %lf) ", local_X.Get(i,j), local_Y.Get(i,j));
        }
        printf ("\n");
      }
    }
    interface_1.Detach();
    interface_2.Detach();
  }
};

#endif

#if SKYLARK_HAVE_COMBBLAS 

template <typename IndexType, 
          typename ValueType>
struct print_t<SpParMat<IndexType,
                        ValueType,
                        SpDCCols<IndexType,ValueType> > > {

  typedef IndexType index_t;
  typedef ValueType value_t;
  typedef SpDCCols<index_t,value_t> seq_matrix_t;
  typedef typename seq_matrix_t::SpColIter seq_matrix_col_iter_t;
  typedef typename seq_matrix_col_iter_t::NzIter seq_matrix_nz_iter_t;
  typedef SpParMat<index_t,value_t, seq_matrix_t> mpi_matrix_t;

  static void apply(const mpi_matrix_t& A,
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;
    if (am_i_printing) printf ("Dump of %s\n", msg);
    seq_matrix_t& data = (const_cast<mpi_matrix_t&>(A)).seq();    
    for(seq_matrix_col_iter_t col=data.begcol();col!=data.endcol();++col) 
      for(seq_matrix_nz_iter_t nz=data.begnz(col);nz!=data.endnz(col);++nz) 
        if (am_i_printing)
          printf ("%d %d %lf\n", 1+col.colid(), 1+nz.rowid(), nz.value());
  }
};

template <typename IndexType, 
          typename ValueType>
struct print_t<FullyDistVec<IndexType,ValueType> > {

  typedef IndexType index_t;
  typedef ValueType value_t;
  typedef FullyDistVec<index_t,value_t> mpi_vector_t;

  static void apply(const mpi_vector_t& x,
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;
    if (am_i_printing) printf ("Dump of %s\n", msg);
    for (index_t i=0; i<x.TotalLength(); ++i) {
      value_t val = x.GetElement(i);
      if (am_i_printing) printf ("%lf\n", val);
    }
  }

  static void apply(const mpi_vector_t& x, 
                    const mpi_vector_t& y, 
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;
    if (am_i_printing) printf ("Dump of %s\n", msg);
    for (index_t i=0; i<x.TotalLength(); ++i) {
      value_t val1 = x.GetElement(i);
      value_t val2 = y.GetElement(i);
      if (am_i_printing) printf ("%lf --- %lf\n", val1, val2);
    }
  }
};

template <typename IndexType, 
          typename ValueType>
struct print_t<FullyDistMultiVec<IndexType,ValueType> > {
  typedef IndexType index_t;
  typedef ValueType value_t;
  typedef FullyDistVec<index_t,value_t> mpi_vector_t;
  typedef FullyDistMultiVec<index_t,value_t> mpi_multi_vector_t;
  typedef print_t<mpi_vector_t> internal_printer_t;

  static void apply(const mpi_multi_vector_t& X,
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;
    if (am_i_printing) printf ("Dump of %s\n", msg);

    const index_t k = X.size;
    const index_t m = X.dim;
    for (index_t i=0; i<m; ++i) {
      for (index_t j=0; j<k; ++j) {
        value_t val = X[j].GetElement(i);
        if (am_i_printing) printf ("%lf ", val);
      }
      if (am_i_printing) printf ("\n");
    }
  }

  static void apply(const mpi_multi_vector_t& X, 
                    const mpi_multi_vector_t& Y, 
                    const char* msg,
                    bool am_i_printing,
                    int debug_level=0) {
    if (1>=debug_level) return;
    if (am_i_printing) printf ("Dump of %s\n", msg);

    const index_t k = X.size;
    const index_t m = X.dim;
    for (index_t i=0; i<m; ++i) {
      for (index_t j=0; j<k; ++j) {
        value_t val_1 = X[j].GetElement(i);
        value_t val_2 = Y[j].GetElement(i);
        if (am_i_printing) printf ("(%lf --- %lf) ", val_1, val_2);
      }
      if (am_i_printing) printf ("\n");
    }
  }
};

#endif

} } /** namespace skylark::nla */

#endif // PRINT_HPP
