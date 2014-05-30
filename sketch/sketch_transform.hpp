#ifndef SKYLARK_SKETCH_TRANSFORMS_HPP
#define SKYLARK_SKETCH_TRANSFORMS_HPP

namespace skylark {
namespace sketch {

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

    virtual const sketch_transform_data_t* get_data() const = 0;

    boost::property_tree::ptree to_ptree() const {
        return get_data()->to_ptree();
    }

    virtual ~sketch_transform_t() {

    }

    static
    sketch_transform_t* from_ptree(const boost::property_tree::ptree& pt);

};

} } /** namespace skylark::sketch */

#endif // SKETCH_TRANSFORMS_HPP
