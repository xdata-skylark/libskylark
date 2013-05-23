/**
 * @file documentation.cpp
 * @author pkambadu
 */
namespace skylark {
namespace python {

const char* module_docstring =
    "The skylark module contains Python wrappers for Skylark's C++ interface.\n"
    "Skylark is a library for randomized numerical linear algebra.\n";

const char* partitioner_docstring =
    "The partitioner class provides helper functions to divide ND spaces \n"
    "into different sized partitions. This is typically used to distribute \n"
    "objects such as matrices when running on a distributed-memory env. \n"
    "There is one partitioner for different type of value types. For \n"
    "example, to divide a space of 32-bit integers, use int32_partitioner\n";

const char* partitioner_divide_docstring =
    "The partitioner.divide function can be used to create 1D sized \n"
    "partitions of a space. Please select a type for the underlying space. \n";

} // namespace python
} // namespace python
