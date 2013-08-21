#ifndef READER_HPP
#define READER_HPP

#include <iterator>
#include <fstream>
#include <string>
#include <sstream>

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

namespace skylark {
namespace utility {

/**
 * A method to read dense matrices within the given range of lines from a
 * text file. The matrix columns/rows which form the lines are assumed to
 * be sorted.
 */
template <typename OutputIterator>
void read_dense_matrix (const char* file_path,
                        int column_start,
                        int column_end,
                        OutputIterator result,
                        const char* separator = " ") {
    typedef typename std::iterator_traits<OutputIterator>::value_type
        value_type;

    std::ifstream in (file_path);
    if (!in.is_open()) {
        printf ("Could not open %s\n", file_path);
        return;
    }

    /**
     * Skip all the lines till we reach the starting point of column_range.
     * Then, read last-first+1 lines and parse them. Quit without processing
     * the rest of the lines.
     */
    std::string current_line;
    int column = 0;
    while ((column_start > column) && (getline (in, current_line))) ++column;

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer_type;
    boost::char_separator<char> separators(separator);

    while ((column_end > column) && (getline (in, current_line))) {
        tokenizer_type tokens(current_line, separators);
        for (tokenizer_type::iterator token_iter = tokens.begin();
             token_iter != tokens.end();
             ++token_iter) {
            *result++ = boost::lexical_cast<value_type>(token_iter->c_str());
        }
        ++column;
    }
}

} // namespace utility
} // namespace skylark

#endif // READER_HPP
