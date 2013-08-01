#ifndef SKETCH_STRATEGIES_HPP
#define SKETCH_STRATEGIES_HPP

namespace skylark {
namespace algorithms {

/** Tags for possible strategies to use sketchs */

/// Base class for tags specifying strategies for using sketches.
struct sketch_use_strategy_tag {};

/// Tag for the sketch and solve smaller approach. (approximate solution)
struct sketch_and_solve_tag : sketch_use_strategy_tag {};

/**
 * Tag for using the sketch to build an accelerator that can be used to solve
 * exactly.
 */
struct sketched_accelerator_tag : sketch_use_strategy_tag {};


} // namespace sketch
} // namespace skylark

#endif // SKETCH_STRATEGIES_HPP
