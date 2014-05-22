#ifndef SKYLARK_TRANSFORMS_HPP
#define SKYLARK_TRANSFORMS_HPP

namespace skylark {
namespace sketch {

/** Define a base class for tagging which dimension you are sketching/transforming */
struct dimension_tag {};

/// Apply the sketch/transform to the columns. In matrix form this is A->SA.
struct columnwise_tag : dimension_tag {};

/// Apply the sketch/transform to the rows. In matrix form this is A->AS^T
struct rowwise_tag : dimension_tag {};

/**
 * Abstract base class for all sketch transforms.
 */
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
class sketch_transform_t {
public:

    virtual void apply (const InputMatrixType& A,
        OutputMatrixType& sketch_of_A, columnwise_tag dimension) const = 0;

    virtual void apply (const InputMatrixType& A,
        OutputMatrixType& sketch_of_A, rowwise_tag dimension) const = 0;

    virtual int get_N() const = 0; /**< Get input dimension */

    virtual int get_S() const = 0; /**< Get output dimension */

    virtual ~sketch_transform_t() {

    }

    static
    sketch_transform_t* from_ptree(const boost::property_tree::ptree& pt);

};

} } /** namespace skylark::sketch */

#endif // TRANSFORMS_HPP
