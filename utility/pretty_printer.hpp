#ifndef SKYLARK_PRETTY_PRINTER_HPP
#define SKYLARK_PRETTY_PRINTER_HPP

#include <stdlib.h>
#include <vector>
#include <iostream>

#define _PP_DEFAULT_MAX_PRINT_ 31
namespace skylark { namespace utility {
/**
  A printing utility for vectors, buffers and
    a row from an El::DistMatrix<El::VC, El::STAR>
  */
template <typename T>
class pretty_printer {
    public:
        static void print_vector(const T* v, size_t len,
                const size_t max_print=_PP_DEFAULT_MAX_PRINT_) {
            std::cout << "[";
            bool trunc = false;
            if (len > max_print) {
                trunc = true;
                len = max_print;
            }
            for (size_t i = 0; i < len; i++)
                std::cout << " " << v[i];
            if (trunc)
                std::cout << " ...";
            std::cout << " ]\n";
        }

        static void print_vector(const std::vector<T>& v,
                const size_t max_print=_PP_DEFAULT_MAX_PRINT_) {
            print_vector(&v[0], v.size(), max_print);
        }

        static void print_matrix(const std::vector<T>& mat,
                const size_t nrow, const size_t ncol,
                const size_t max_print=_PP_DEFAULT_MAX_PRINT_) {
            print_matrix(&mat[0], nrow, ncol, max_print);
        }

        static void print_matrix(const T* mat,
                const size_t nrow, const size_t ncol,
                const size_t max_print=_PP_DEFAULT_MAX_PRINT_) {
            for (size_t row = 0; row < nrow; row++)
                print(&mat[row*ncol], ncol, max_print);
        }

        static void print_vc_star_row(const T* buf, const size_t width,
                const size_t height, const std::string msg,
                const size_t max_print=_PP_DEFAULT_MAX_PRINT_) {
            std::string str = msg;
            str += std::string("[ ");

            size_t numprocd = 0;
            while (numprocd < width)
                str += std::to_string(buf[numprocd++*height])
                    + std::string(" ");

            str += std::string("]");
            std::cout << str << std::endl;
        }
};
} }  // namespace skylark::utility
#endif // SKYLARK_PRETTY_PRINTER_HPP
