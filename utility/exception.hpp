#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

#include <string>
#include <exception>

#include <boost/exception/all.hpp>

const char* const errmsg[] = {
      "Skylark failure"
    , "Skylark failed while communicating (MPI)"
    , "Skylark failed in a call to Elemental"
    , "Skylark failed due to a unsupported matrix distribution"
    , "Skylark failed in a call to CombBLAS"
    , "Skylark failed in sketching a matrix"
    , "Skylark failed when allocating memory in a sketch"
};

/// resolves an error_code to a human readable failure message
const char* skylark_strerror(int error_code) {
    return errmsg[error_code - 100];
}


/// macro defining the beginning of a try block
#define SKYLARK_BEGIN_TRY() try {
/// macro defining the end of a try block
#define SKYLARK_END_TRY()   }

/// throw a skylark exception including file/line information
#define SKYLARK_THROW_EXCEPTION(x) \
    BOOST_THROW_EXCEPTION(x);

/// print exception details to stderr
#define SKYLARK_PRINT_EXCEPTION_DETAILS(ex) \
    std::cerr << boost::diagnostic_information(ex) << std::endl;

/// print the exception trace (if available) to stderr
#define SKYLARK_PRINT_EXCEPTION_TRACE(ex) \
    if (const std::string *trace = \
            boost::get_error_info<skylark::utility::stack_trace>(ex)) { \
        std::cerr << *trace << std::endl; \
    }

/// catch a Skylark exceptions and returns an error code
//XXX: only get top-most exception?
#define SKYLARK_CATCH_AND_RETURN_ERROR_CODE() \
    catch (const skylark::utility::skylark_exception& ex) { \
        if (int const *c = \
                boost::get_error_info<skylark::utility::error_code>(ex)) { \
            return *c; \
        } \
    }



namespace skylark {
namespace utility {

/// predefined structure for error code
typedef boost::error_info<struct tag_error_code, int>         error_code;
/// predefined structure for error msg
typedef boost::error_info<struct tag_error_msg,  std::string> error_msg;

/// predefined structure for trace error (appends messages)
typedef boost::error_info<struct tag_stack_trace,  std::string> stack_trace;
typedef boost::error_info<struct tag_append_trace, std::string> append_trace;

/// define a base exception
struct skylark_exception : virtual boost::exception, virtual std::exception {

    skylark_exception() {
        *this << error_code(100);
    }

    skylark_exception& operator<< (const append_trace& rhs) {

        std::string trace_value = "";

        if( const std::string *cur_trace =
            boost::get_error_info<stack_trace, skylark_exception>(*this) ) {
            trace_value.append(*cur_trace);
            trace_value.append("\n");
        }

        trace_value.append(rhs.value());
        *this << stack_trace(trace_value);
        return *this;
    }

};

/// exceptions thrown by Elemental
struct elemental_exception : virtual skylark_exception {
public:
    using skylark_exception::operator<<;

    elemental_exception() {
        *this << error_code(102);
    }
};
/// exceptions thrown by CombBLAS
struct combblas_exception : virtual skylark_exception {
public:
    using skylark_exception::operator<<;

    combblas_exception() {
        *this << error_code(104);
    }
};
/// exceptions thrown by Boost MPI
struct mpi_exception : virtual skylark_exception {
public:
    using skylark_exception::operator<<;

    mpi_exception() {
        *this << error_code(101);
    }
};

/// exceptions in the sketch layer
struct sketch_exception : virtual skylark_exception {
public:
    using skylark_exception::operator<<;

    sketch_exception() {
        *this << error_code(105);
    }
};

/// exceptions for allocating memory in sketch layer
struct allocation_exception : virtual sketch_exception  {
public:
    using sketch_exception::operator<<;

    allocation_exception() {
        *this << error_code(106);
    }
};

/// exceptions for unsupported matrix distributions in Elemental
struct unsupported_matrix_distribution : virtual elemental_exception {
public:
    using elemental_exception::operator<<;

    unsupported_matrix_distribution() {
        *this << error_code(103);
    }
};

} // namespace utility
} // namespace skylark

#endif
