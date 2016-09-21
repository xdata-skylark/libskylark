#include <stdlib.h>
#include <iostream>
#include <skylark.hpp>

namespace skyutil = skylark::utility;

int main(int argc, char* argv[]) {
#if 1
    double eigsarr[] = {
        15.499461, 14.577127, 14.535973, 14.465264, 14.457898, 11.924074,
        10.411152, 9.637591, 7.665476, 7.496989, 7.465672, 7.414213,
        6.499588, 6.241150, 5.679422, 5.493338, 5.123712, 5.098758,
        5.067389, 5.051121, 5.039233, 5.026632, 5.012424, 5.009729,
        5.001122, 4.987087, 4.983646, 4.980309, 4.975617, 4.962617,
        4.954966, 4.949616, 4.944045, 4.940561, 4.939032, 4.930906,
        4.929919, 4.924148, 4.922182, 4.921777, 4.916565, 4.905751,
        4.905021, 4.900150, 4.897464, 4.896930, 4.894344, 4.885108,
        4.871480, 4.866175};
#else
    double eigsarr[] = { 2.72001, 1.93008, 1.88029, 1.81426, 1.73366, 1.57313 };
#endif

    std::vector<double> eigs(eigsarr, eigsarr +
            (sizeof eigsarr / sizeof eigsarr[0]));

    std::vector<unsigned> elbs = skyutil::get_elbows(eigs);
    std::cout << "[ ";
    for (std::vector<unsigned>::iterator it = elbs.begin(); it != elbs.end();
            ++it)
        std::cout << *it << " ";
    std::cout << "]\n";

    return EXIT_SUCCESS;
}
