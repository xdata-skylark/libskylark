#ifndef SKYLARK_SKETCH_TRANSFORMS_HPP
#define SKYLARK_SKETCH_TRANSFORMS_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

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

    /**
     * Return a type erased version of the current transform.
     * Will allocate a new object!
     */
    sketch_transform_t<boost::any, boost::any> *type_erased() const {
        return get_data()->get_transform();
    }

    virtual ~sketch_transform_t() {

    }

    static
    sketch_transform_t* from_ptree(const boost::property_tree::ptree& pt);
};

/**
 * Specialze for the boost::any input/output.
 * Note: the convention is to pass pointers in the apply!
 */
template<>
class sketch_transform_t<boost::any, boost::any> {
public:

    virtual void apply (const boost::any& A,
        const boost::any& sketch_of_A, columnwise_tag dimension) const = 0;

    virtual void apply (const boost::any& A,
        const boost::any& sketch_of_A, rowwise_tag dimension) const = 0;

    virtual int get_N() const = 0; /**< Get input dimension */

    virtual int get_S() const = 0; /**< Get output dimension */

    virtual const sketch_transform_data_t* get_data() const = 0;

    boost::property_tree::ptree to_ptree() const {
        return get_data()->to_ptree();
    }

    /**
     * Return a type erased version of the current transform.
     * Will allocate a new object!
     */
    sketch_transform_t<boost::any, boost::any> *type_erased() const {
        return get_data()->get_transform();
    }

    virtual ~sketch_transform_t() {

    }

    static
    sketch_transform_t* from_ptree(const boost::property_tree::ptree& pt);

};

/** Short types name for use in macros */
namespace mdtypes {

typedef El::Matrix<double> matrix_t;
typedef base::sparse_matrix_t<double> sparse_matrix_t;
typedef El::DistMatrix<double> dist_matrix_t;
typedef El::DistMatrix<double, El::STAR, El::STAR> shared_matrix_t;
typedef El::DistMatrix<double, El::CIRC, El::CIRC> root_matrix_t;
typedef El::DistMatrix<double, El::VC, El::STAR> dist_matrix_vc_star_t;
typedef El::DistMatrix<double, El::VR, El::STAR> dist_matrix_vr_star_t;
typedef El::DistMatrix<double, El::STAR, El::VC> dist_matrix_star_vc_t;
typedef El::DistMatrix<double, El::STAR, El::VR> dist_matrix_star_vr_t;

// TODO
//#ifdef SKYLARK_HAVE_COMBBLAS
//typedef SpParMat<size_t, double, SpDCCols<size_t, double> >
//cb_dist_sparse_matrix_t;
//#endif

}

namespace mftypes {

typedef El::Matrix<float> matrix_t;
typedef base::sparse_matrix_t<float> sparse_matrix_t;
typedef El::DistMatrix<float> dist_matrix_t;
typedef El::DistMatrix<float, El::STAR, El::STAR> shared_matrix_t;
typedef El::DistMatrix<float, El::CIRC, El::CIRC> root_matrix_t;
typedef El::DistMatrix<float, El::VC, El::STAR> dist_matrix_vc_star_t;
typedef El::DistMatrix<float, El::VR, El::STAR> dist_matrix_vr_star_t;
typedef El::DistMatrix<float, El::STAR, El::VC> dist_matrix_star_vc_t;
typedef El::DistMatrix<float, El::STAR, El::VR> dist_matrix_star_vr_t;

// TODO
//#ifdef SKYLARK_HAVE_COMBBLAS
//typedef SpParMat<size_t, float, SpDCCols<size_t, float> >
//cb_dist_sparse_matrix_t;
//#endif

}

#define SKYLARK_SKETCH_ANY_APPLY_DISPATCH(I, O, C)                      \
    if (sketch_of_A.type() == typeid(O*))  {                            \
        if (A.type() == typeid(I*)) {                                   \
            C<I, O > S(*this);                                          \
            S.apply(*boost::any_cast<I *>(A),                           \
                *boost::any_cast<O*>(sketch_of_A), dimension);          \
            return;                                                     \
        }                                                               \
        if (A.type() == typeid(const I*)) {                             \
            C<I, O > S(*this);                                          \
            S.apply(*boost::any_cast<const I *>(A),                     \
                *boost::any_cast<O*>(sketch_of_A), dimension);          \
            return;                                                     \
        }                                                               \
    }

#define SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(I, O, C)                  \
    if (sketch_of_A.type() == typeid(O*))  {                            \
        if (A.type() == typeid(I*)) {                                   \
            C<I, O, QMCSequenceType> S(*this);                          \
            S.apply(*boost::any_cast<I *>(A),                           \
                *boost::any_cast<O*>(sketch_of_A), dimension);          \
            return;                                                     \
        }                                                               \
        if (A.type() == typeid(const I*)) {                             \
            C<I, O, QMCSequenceType> S(*this);                          \
            S.apply(*boost::any_cast<const I *>(A),                     \
                *boost::any_cast<O*>(sketch_of_A), dimension);          \
            return;                                                     \
        }                                                               \
    }

} } /** namespace skylark::sketch */

#endif // SKETCH_TRANSFORMS_HPP
