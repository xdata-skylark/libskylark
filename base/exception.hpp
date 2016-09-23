#ifndef SKYLARK_EXCEPTION_HPP
#define SKYLARK_EXCEPTION_HPP

#include <boost/exception/all.hpp>

#include <exception>
#include <string>

const char* const skylark_errmsg[] = {
    "Skylark failure"
    , "Skylark failed while communicating (MPI)"
    , "Skylark failed in a call to Elemental"
    , "Skylark failed due to a unsupported matrix distribution"
    , "Skylark failed in a call to CombBLAS"
    , "Skylark failed in sketching layer"
    , "Skylark failed in nla operation"
    , "Skylark failed when allocating memory in a sketch"
    , "Skylark failed in a call into the Random123 layer"
    , "Skylark failed due to a unsupported base operation"
    , "Skylark failed in I/O calls"
    , "Skylark failed because invalid parameter was passed"
    , "Skylark failed because base classes were used incorrectly"
    , "Skylark failed in ml operation"
};

/// resolves an error_code to a human readable failure message
const char* skylark_strerror(int error_code) {
    return skylark_errmsg[error_code - 100];
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
            boost::get_error_info<skylark::base::stack_trace>(ex)) { \
        std::cerr << *trace << std::endl; \
    }

// catch and print message
#define SKYLARK_CATCH_AND_PRINT(p) \
    catch (const skylark::base::skylark_exception& ex) { \
        if (p == true) SKYLARK_PRINT_EXCEPTION_DETAILS(ex); \
    } \

/// catch a Skylark exception and returns an error code
#define SKYLARK_CATCH_AND_RETURN_ERROR_CODE() \
    catch (const skylark::base::skylark_exception& ex) { \
        if (int const *c = \
                boost::get_error_info<skylark::base::error_code>(ex)) { \
            return *c; \
        } \
    }

/// catch a Skylark exception, keep a pointer to it, and returns an error code
#define SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(ptr) \
    catch (const skylark::base::skylark_exception& ex) { \
        ptr = boost::current_exception();                \
        if (int const *c = \
                boost::get_error_info<skylark::base::error_code>(ex)) { \
            return *c; \
        } \
    }

namespace skylark {
namespace base {

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

        if (const std::string *cur_trace =
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

/// exceptions in the nla layer
struct nla_exception : virtual skylark_exception {
 public:
    using skylark_exception::operator<<;

    nla_exception() {
        *this << error_code(106);
    }
};

/// exceptions in the ml layer
struct ml_exception : virtual skylark_exception {
 public:
    using skylark_exception::operator<<;

    ml_exception() {
        *this << error_code(113);
    }
};

/// exceptions in the Random123 layer
struct random123_exception : virtual skylark_exception {
 public:
    using skylark_exception::operator<<;

    random123_exception() {
        *this << error_code(108);
    }
};

/// exceptions when doing I/O
struct io_exception : virtual skylark_exception {
 public:
    using skylark_exception::operator<<;

    io_exception() {
        *this << error_code(110);
    }
};

/// exceptions for allocating memory in sketch layer
struct allocation_exception : virtual sketch_exception  {
 public:
    using sketch_exception::operator<<;

    allocation_exception() {
        *this << error_code(107);
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

/// exceptions for unsupported base operation
struct unsupported_base_operation : virtual elemental_exception {
 public:
    using skylark_exception::operator<<;

    unsupported_base_operation() {
        *this << error_code(109);
    }
};

/// exceptions for invalid parameters passed
struct invalid_parameters : virtual skylark_exception {
 public:
    using skylark_exception::operator<<;

    invalid_parameters() {
        *this << error_code(111);
    }
};

/// exceptions for invalid usage of base classes (runtime check that detects
/// bugs)
struct invalid_usage : virtual skylark_exception {
 public:
    using skylark_exception::operator<<;

    invalid_usage() {
        *this << error_code(112);
    }
};

}  // namespace base
}  // namespace skylark

#endif  // SKYLARK_EXCEPTION_HPP
