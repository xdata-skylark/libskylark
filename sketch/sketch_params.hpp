#ifndef SKYLARK_SKETCH_PARAMS_HPP
#define SKYLARK_SKETCH_PARAMS_HPP


namespace skylark { namespace sketch {

/** Should be changed by the user to a nonzero value to indicate
 *  the intention NOT to use maximum blocksize value.
*/
int blocksize = 0;

double factor = 20.;

void set_blocksize(int blocksize) {
    skylark::sketch::blocksize = blocksize;
}

int get_blocksize() {
    return skylark::sketch::blocksize;
}

void set_factor(double factor) {
    skylark::sketch::factor = factor;
}

double get_factor() {
    return skylark::sketch::factor;
}

} } /** namespace skylark::sketch */

#endif // SKYLARK_SKETCH_PARAMS_HPP
