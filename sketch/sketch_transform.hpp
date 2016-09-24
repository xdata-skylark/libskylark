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
struct sketch_transform_t {

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
struct sketch_transform_t<boost::any, boost::any> {

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

typedef sketch_transform_t<boost::any, boost::any> generic_sketch_transform_t;
typedef std::shared_ptr<generic_sketch_transform_t>  generic_sketch_transform_ptr_t;

/**
 * Container of sketch_transform_t<any, any> that makes the input/output
 * types concrete.
 */
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct sketch_transform_container_t 
    : public sketch_transform_t<InputMatrixType, OutputMatrixType> {

    sketch_transform_container_t() : _transform(nullptr) {

    }

    sketch_transform_container_t(const generic_sketch_transform_ptr_t &transform)
        : _transform(transform) {

    }

    sketch_transform_container_t(const sketch_transform_container_t &other)
        : _transform(other._transform) {

    }

    sketch_transform_container_t operator=(const sketch_transform_container_t &other) {
        if (this != &other)
            _transform = other._transform;

        return *this;
    }

    virtual void apply (const InputMatrixType& A,
        OutputMatrixType& sketch_of_A, columnwise_tag dimension) const {
        _transform->apply(&A, &sketch_of_A, dimension);
    }

    virtual void apply (const InputMatrixType& A,
        OutputMatrixType& sketch_of_A, rowwise_tag dimension) const {
        _transform->apply(&A, &sketch_of_A, dimension);
    }

    virtual int get_N() const {
        return _transform->get_N();
    }

    virtual int get_S() const {
        return _transform->get_S();
    }

    virtual const sketch_transform_data_t* get_data() const {
        return _transform->get_data();
    }


    virtual ~sketch_transform_container_t() {

    }

    static
    sketch_transform_container_t from_ptree(const boost::property_tree::ptree& pt) {
        generic_sketch_transform_ptr_t 
            s_ptr(generic_sketch_transform_t::from_ptree(pt));
        return sketch_transform_container_t(s_ptr);
    }

    bool empty() const {
        return _transform == nullptr;

    }

private:
    generic_sketch_transform_ptr_t _transform;

};

/**
 * Specialization for ::any
 */
template<>
struct sketch_transform_container_t<boost::any, boost::any> 
    : public sketch_transform_t<boost::any, boost::any> {

    sketch_transform_container_t() : _transform(nullptr) {

    }

    sketch_transform_container_t(const generic_sketch_transform_ptr_t &transform)
        : _transform(transform) {

    }

    sketch_transform_container_t(const sketch_transform_container_t &other)
        : _transform(other._transform) {

    }

    sketch_transform_container_t operator=(const sketch_transform_container_t &other) {
        if (this != &other)
            _transform = other._transform;

        return *this;
    }

    virtual void apply (const boost::any& A,
        const boost::any& sketch_of_A, columnwise_tag dimension) const {
        _transform->apply(A, sketch_of_A, dimension);
    }

    virtual void apply (const boost::any& A,
        const boost::any& sketch_of_A, rowwise_tag dimension) const {
        _transform->apply(A, sketch_of_A, dimension);
    }

    virtual int get_N() const {
        return _transform->get_N();
    }

    virtual int get_S() const {
        return _transform->get_S();
    }

    virtual const sketch_transform_data_t* get_data() const {
        return _transform->get_data();
    }


    virtual ~sketch_transform_container_t() {

    }

    static
    sketch_transform_container_t from_ptree(const boost::property_tree::ptree& pt) {
        generic_sketch_transform_ptr_t 
            s_ptr(generic_sketch_transform_t::from_ptree(pt));
        return sketch_transform_container_t(s_ptr);
    }

    bool empty() const {
        return _transform == nullptr;

    }

private:
    generic_sketch_transform_ptr_t _transform;

};

typedef sketch_transform_container_t<boost::any, boost::any>
generic_sketch_container_t;

/**
 * Creating generic skethes
 */
template<template <typename, typename> class SketchType>
generic_sketch_container_t create_sketch(El::Int N, El::Int S,
    const typename SketchType<boost::any, boost::any>::params_t &params,
    base::context_t &context) {

    generic_sketch_transform_ptr_t
        S0(new SketchType<boost::any, boost::any>(N, S, params, context));
    return generic_sketch_container_t(S0);
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
