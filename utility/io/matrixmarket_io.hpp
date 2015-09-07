/*
 * matrixmarket_io.hpp
 *
 *  Created on: Aug 8, 2015
 *      Author: chander
 */

#ifndef MATRIXMARKET_IO_HPP_
#define MATRIXMARKET_IO_HPP_

namespace skylark { namespace utility { namespace io {

template<typename T>
inline void ReadMatrixMarket(El::DistSparseMatrix<T>& A, const std::string filename) {
//        DEBUG_ONLY(El::CallStackEntry cse("read::MatrixMarket"))
        typedef El::Base<T> Real ;
        std::ifstream file(filename.c_str());
        if (!file.is_open())
                El::RuntimeError("Could not open ", filename);
// Read the header
// ===============
// Attempt to pull in the various header components
// ------------------------------------------------
        std::string line, stamp, object, format, field, symmetry;
        if (!std::getline(file, line))
                El::RuntimeError("Could not extract header line");
        {
                std::stringstream lineStream(line);
                lineStream >> stamp;
                if (stamp != std::string("%%MatrixMarket"))
                        El::RuntimeError("Invalid Matrix Market stamp: ", stamp);
                if (!(lineStream >> object))
                        El::RuntimeError("Missing Matrix Market object");
                if (!(lineStream >> format))
                        El::RuntimeError("Missing Matrix Market format");
                if (!(lineStream >> field))
                        El::RuntimeError("Missing Matrix Market field");
                if (!(lineStream >> symmetry))
                        El::RuntimeError("Missing Matrix Market symmetry");
        }
        // Ensure that the header components are individually valid
        // --------------------------------------------------------
                const bool isMatrix = (object == std::string("matrix"));
                const bool isArray = (format == std::string("array"));
                const bool isComplex = (field == std::string("complex"));
                const bool isPattern = (field == std::string("pattern"));
                const bool isGeneral = (symmetry == std::string("general"));
                const bool isSymmetric = (symmetry == std::string("symmetric"));
                const bool isSkewSymmetric = (symmetry == std::string("skew-symmetric"));
                const bool isHermitian = (symmetry == std::string("hermitian"));
                if (!isMatrix && object != std::string("vector"))
                        El::RuntimeError("Invalid Matrix Market object: ", object);
                if (!isArray && format != std::string("coordinate"))
                        El::RuntimeError("Invalid Matrix Market format: ", format);
                if (!isComplex && !isPattern && field != std::string("real")
                                && field != std::string("double")
                                && field != std::string("integer"))
                        El::RuntimeError("Invalid Matrix Market field: ", field);
                if (!isGeneral && !isSymmetric && !isSkewSymmetric && !isHermitian)
                        El::RuntimeError("Invalid Matrix Market symmetry: ", symmetry);
        // Ensure that the components are consistent
        // -----------------------------------------
                if (isArray && isPattern)
                        El::RuntimeError("Pattern field requires coordinate format");
        // NOTE: This constraint is only enforced because of the note located at
        // http://people.sc.fsu.edu/~jburkardt/data/mm/mm.html
                if (isSkewSymmetric && isPattern)
                        El::RuntimeError("Pattern field incompatible with skew-symmetry");
                if (isHermitian && !isComplex)
                        El::RuntimeError("Hermitian symmetry requires complex data");
        // Skip the comment lines
        // ======================
                while (file.peek() == '%')
                        std::getline(file, line);
                int m, n;
                if (!std::getline(file, line))
                        El::RuntimeError("Could not extract the size line");
                if (isArray) {
        // Read in the matrix dimensions
        // =============================
                    if (isMatrix) {
                            std::stringstream lineStream(line);
                            if (!(lineStream >> m))
                                    El::RuntimeError("Missing matrix height: ", line);
                            if (!(lineStream >> n))
                                    El::RuntimeError("Missing matrix width: ", line);
                    } else {
                            std::stringstream lineStream(line);
                            if (!(lineStream >> m))
                                    El::RuntimeError("Missing vector height: ", line);
                            n = 1;
                    }
    // Resize the matrix
    // =================
                    A.Resize(m, n);
    // Now read in the data
    // ====================
                    Real realPart, imagPart;
                    for (El::Int j = 0; j < n; ++j) {
                            for (El::Int i = 0; i < m; ++i) {
                                    if (!std::getline(file, line))
                                            El::RuntimeError("Could not get entry (", i, ",", j, ")");
                                    std::stringstream lineStream(line);
                                    if (!(lineStream >> realPart))
                                            El::RuntimeError("Could not extract real part of entry (", i,
                                                            ",", j, ")");
                                    A.QueueUpdate(i, j, realPart);
    /*                              if (isComplex) {
                                            if (!(lineStream >> imagPart))
                                                    El::RuntimeError("Could not extract imag part of entry (",
                                                                    i, ",", j, ")");
                                            A.SetImagPart(i, j, imagPart);
                                    }
    */                      }
                    }
            } else {
            	// Read in the matrix dimensions and number of nonzeros
            	// ====================================================
            	                int numNonzero;
            	                if (isMatrix) {
            	                        std::stringstream lineStream(line);
            	                        if (!(lineStream >> m))
            	                                El::RuntimeError("Missing matrix height: ", line);
            	                        if (!(lineStream >> n))
            	                                El::RuntimeError("Missing matrix width: ", line);
            	                        if (!(lineStream >> numNonzero))
            	                                El::RuntimeError("Missing nonzeros entry: ", line);
            	                } else {
            	                        std::stringstream lineStream(line);
            	                        if (!(lineStream >> m))
            	                                El::RuntimeError("Missing vector height: ", line);
            	                        n = 1;
            	                        if (!(lineStream >> numNonzero))
            	                                El::RuntimeError("Missing nonzeros entry: ", line);
            	                }
            	// Create a matrix of zeros
            	// ========================
            	                A.Resize(m, n);
            	                A.Reserve( numNonzero );
            	// Fill in the nonzero entries
            	// ===========================
            	                int i, j;
            	                Real realPart, imagPart;
            	                for (El::Int k = 0; k < numNonzero; ++k) {
            	                        if (!std::getline(file, line))
            	                                El::RuntimeError("Could not get nonzero ", k);
            	                        std::stringstream lineStream(line);
            	                        if (!(lineStream >> i))
            	                                El::RuntimeError("Could not extract row coordinate of nonzero ", k);
            	                        --i; // convert from Fortran to C indexing
            	                        if (isMatrix) {
            	                                if (!(lineStream >> j))
            	                                        El::RuntimeError("Could not extract col coordinate of nonzero ",
            	                                                        k);
            	                                --j;
            	                        } else
                                            j = 0;
                                    if (isPattern) {
                                            A.QueueUpdate(i, j, T(1));
                                    } else {
                                            if (!(lineStream >> realPart))
                                                    El::RuntimeError("Could not extract real part of entry (", i,
                                                                    ",", j, ")");
                                            if( i >= A.FirstLocalRow() && i < (A.FirstLocalRow()+A.LocalHeight() ) )
                                                    A.QueueLocalUpdate( i-A.FirstLocalRow(), j, realPart );
            //                              A.Update(i, j, realPart);
            /*                              if (isComplex) {
                                                    if (!(lineStream >> imagPart))
                                                            El::RuntimeError("Could not extract imag part of entry (",
                                                                            i, ",", j, ")");
                                                    A.UpdateImagPart(i, j, imagPart);
                                            }
            */                      }
                            }
                            A.MakeConsistent();
                    }
                    if (isSymmetric)
                            MakeSymmetric(El::UpperOrLowerNS::LOWER, A);
                    if (isHermitian)
                            MakeHermitian(El::UpperOrLowerNS::LOWER, A);
            // I'm not certain of what the MM standard is for complex skew-symmetry,
            // so I'll default to assuming no conjugation
                    const bool conjugateSkew = false;
                    if (isSkewSymmetric) {
                            MakeSymmetric(El::UpperOrLowerNS::LOWER, A, conjugateSkew);
                            ScaleTrapezoid(T(-1), El::UpperOrLowerNS::UPPER, A, 1);
                    }
            }


            } } } // namespace skylark::utility::io

#endif /* MATRIXMARKET_IO_HPP_ */
