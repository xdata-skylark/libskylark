#ifndef SKETCH_STRATEGIES_HPP
#define SKETCH_STRATEGIES_HPP

namespace skylark {
namespace algorithms {

/** Tags for possible strategies to use sketchs */

struct sketch_use_strategy_tag {};

struct sketch_and_solve_tag : sketch_use_strategy_tag {};

struct sketched_accelerator : sketch_use_strategy_tag {};


} // namespace sketch
} // namespace skylark
 
#endif // SKETCH_STRATEGIES_HPP
