#ifndef SKYLARK_SKETCH_PARAMS_HPP
#define SKYLARK_SKETCH_PARAMS_HPP


namespace skylark { namespace sketch {

/**
 * Set value to 0 to force no-blocking in sketching.
 * Better performance, but much more memory.
 * Current value, will lower memory usage.
 *
 */

namespace params {

int blocksize = 1000;

double factor = 20.;

}

void set_blocksize(int blocksize) {
    params::blocksize = blocksize;
}

int get_blocksize() {
    return params::blocksize;
}

void set_factor(double factor) {
    params::factor = factor;
}

double get_factor() {
    return params::factor;
}

} } /** namespace skylark::sketch */

#endif // SKYLARK_SKETCH_PARAMS_HPP
