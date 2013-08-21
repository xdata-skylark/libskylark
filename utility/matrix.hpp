#ifndef MATRIX_HPP
#define MATRIX_HPP

namespace skylark {
namespace utility {

/** Define a base type for describing the storage order of the matrix */
struct storage_order_tag {};

/** This structure is used to denote row major storage of the matrix */
struct row_major_tag : storage_order_tag {};

/** This structure is used to denote col major storage of the matrix */
struct col_major_tag : storage_order_tag {};

/** Define a base type to describe the distribution of the matrix */
struct distribution_tag {};

/** This structure is used to denote 1D block distribution of the matrix */
struct one_D_tag : distribution_tag {};

/** This structure is used to denote 1D cyclic distribution of the matrix */
struct two_D_tag : distribution_tag {};

/** This structure is used to figure out is the matrix is sparse or dense */
struct sparsity_tag {};

/** This structure is used to denote that a matrix is dense */
struct dense_tag : sparsity_tag {};

/** This structure is used to denote that a matrix is sparse */
struct sparse_tag : sparsity_tag {};

} // namespace utility
} // namespace skylark

#endif // MATRIX_HPP
