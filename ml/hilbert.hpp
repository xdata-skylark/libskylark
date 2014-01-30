/*
 * hilbert.hpp
 *
 *  Created on: Jan 15, 2014
 *      Author: vikas
 */

#ifndef HILBERT_HPP_
#define HILBERT_HPP_

#include "FunctionProx.hpp"
#include "FeatureTransform.hpp"
#include "BlockADMM.hpp"
#include "options.hpp"

enum LossType {SQUARED, LAD, HINGE, LOGISTIC};
enum RegularizerType {L2, L1};
enum ProblemType {REGRESSION, CLASSIFICATION};
enum KernelType {RBF};

#endif /* HILBERT_HPP_ */
