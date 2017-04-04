#ifndef SKYLARK_KERNELS_HPP
#define SKYLARK_KERNELS_HPP

#include "../../sketch/sketch.hpp"
#include "../feature_transform_tags.hpp"


// Base class for kernels
#include "BaseKernel.hpp"

// Generic gram function for kernels
#include "gram.hpp"

// Skylark kernels
#include "LinearKernel.hpp"
#include "MaternKernel.hpp"
#include "GaussianKernel.hpp"
#include "LaplacianKernel.hpp"
#include "PolynomialKernel.hpp"
#include "ExpsemigroupKernel.hpp"

// Generic Container for kernels
#include "KernelContainer.hpp"


#endif // SKYLARK_KERNELS_HPP
