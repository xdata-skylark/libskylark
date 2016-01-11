#ifndef SKYLARK_SPECTRAL_EMBEDDINGS_HPP
#define SKYLARK_SPECTRAL_EMBEDDINGS_HPP

namespace skylark { namespace ml {

/**
 * Parameter structure for approximate ASE: essentially the same
 * as the structure for SVD (these are all the parameters we need to
 * control).
 */
struct approximate_ase_params_t : public nla::approximate_svd_params_t {

};

namespace internal {

template<typename GraphType, typename AdjacencyType, typename EmbeddingsType,
         typename SType>
void TemplatedApproximateASE(const GraphType& G, int k,
    std::vector<typename GraphType::vertex_type> &indexmap,
    EmbeddingsType &X,
    base::context_t &context, approximate_ase_params_t &params) {

    // Get adjacency matrix.
    AdjacencyType A;
    G.adjacency_matrix(A, indexmap);

    // Compute SVD
    SType S;
    nla::ApproximateSymmetricSVD(El::LOWER,
        A, X, S, k, context, params);

    // Scale columns
    for(El::Int i = 0; i < S.Height(); i++)
        S.Set(i, 0, std::sqrt(S.Get(i, 0)));
    El::DiagonalScale(El::RIGHT, El::NORMAL, S, X);
}

}

/**
 * Approximate Adjacency Spectral Embeddings (ASE).
 *
 * Based on the description of ASE in
 * "Community Detection and Classification in Hierarchical Stochastic Blockmodels"
 * by Lyzinski et al.
 *
 * @tparam GraphType type of graph object. Needs to support the following:
 *                   GraphType::vertex_type - type of vertex.
 *                   GraphType::adjancy_matrix - fill the adjancy matrix.
 * @param G input graph
 * @param k dimension of the embeddings.
 * @param X the computed spectral embeddings, in matrix form.
 * @param indexmap A map from row index in X to vertex. Filled by the function.
 * @param context skylark context to use.
 * @param params parameters.
 */
template<typename GraphType, typename T>
void ApproximateASE(const GraphType& G, int k,
    std::vector<typename GraphType::vertex_type> &indexmap,
    El::Matrix<T> &X, base::context_t &context,
    approximate_ase_params_t params = approximate_ase_params_t()) {

    typedef typename GraphType::vertex_type vertex_type;
    typedef base::sparse_matrix_t<T> adjacency_type;
    typedef El::Matrix<T> embeddings_type;
    typedef El::Matrix<T> S_type;

    internal::TemplatedApproximateASE<GraphType, adjacency_type,
                                      embeddings_type, S_type>
        (G, k, indexmap, X, context, params);

}

template<typename GraphType, typename T>
void ApproximateASE(const GraphType& G, int k,
    std::vector<typename GraphType::vertex_type> &indexmap,
    El::DistMatrix<T, El::VC, El::STAR> &X, base::context_t &context,
    approximate_ase_params_t params = approximate_ase_params_t()) {

    typedef typename GraphType::vertex_type vertex_type;
    typedef base::sparse_vc_star_matrix_t<T> adjacency_type;
    typedef El::DistMatrix<T, El::VC, El::STAR> embeddings_type;
    typedef El::DistMatrix<T, El::STAR, El::STAR> S_type;

    internal::TemplatedApproximateASE<GraphType, adjacency_type,
                                      embeddings_type, S_type>
        (G, k, indexmap, X, context, params);

}

} }   // namespace skylark::ml

#endif // SKYLARK_SPECTRAL_EMBEDDINGS_HPP
