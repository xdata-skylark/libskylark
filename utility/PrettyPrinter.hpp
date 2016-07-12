#ifndef __SKYLARK_PRETTY_PRINTER_HPP__
#define __SKYLARK_PRETTY_PRINTER_HPP__

#include "El.hpp"

namespace skylark {
    namespace utility {

        /**
          A printing utility for vectors, buffers and
            a row from an El::DistMatrix<El::VC, El::STAR>
          */
        template <typename T>
            class PrettyPrinter
            {
                public:
                    static void print(const T* v, const size_t len)
                    {
                        std::cout << "[";
                        for (size_t i = 0; i < len; i++)
                        {
                            std::cout << " " << v[i];
                        }
                        std::cout << " ]\n";
                    }

                    static void print(const std::vector<T>& v)
                    {
                        print(&v[0], v.size());
                    }

                    static void print(const std::vector<T>& mat,
                            const size_t nrow, const size_t ncol)
                    {
                        print(&mat[0], nrow, ncol);
                    }

                    static void print(const T* mat,
                            const size_t nrow, const size_t ncol)
                    {
                        for (size_t row = 0; row < nrow; row++)
                        {
                            print(&mat[row*ncol], ncol);
                        }
                    }

                    static void print_vc_star_row(const T* buf, const size_t width,
                            const size_t height, const std::string msg)
                    {
                        std::string str = msg;
                        str += std::string("[ ");

                        size_t numprocd = 0;
                        while (numprocd < width)
                            str += std::to_string(buf[numprocd++*height]) + std::string(" ");

                        str += std::string("]");
                        El::Output(str);
                    }
            };

    } }  // namespace skylark::utility
#endif // SKYLARK_PRETTY_PRINTER_HPP
