#ifndef SKYLARK_KMEANS_HPP
#define SKYLARK_KMEANS_HPP

#include <cassert>
#include <skylark.hpp>

#define root 0
#define KM_DEBUG 0
#define INVALID_ID UINT_MAX

/*******************************************/
namespace skyutil = skylark::utility;
/*******************************************/

// Annymous namespace for use only here
namespace {
enum init_t {
    RANDOM,
    FORGY,
    PLUSPLUS,
    BARBAR,
    SKETCH,
    NONE
};

/**
 * Given a matrix of n X m
 */
template <typename T>
void row_mean(const El::Matrix<T>& from_mat,
        El::Matrix<T>& to_mat, const El::Matrix<El::Int>& counts) {
    assert(counts.Width() == to_mat.Height());
    // Division by row
    for (El::Unsigned cl = 0; cl < to_mat.Height(); cl++) {
        for (El::Unsigned col = 0; col < to_mat.Width(); col++) {
            // Update global clusters
            if (counts.Get(0, cl) > 0)
                to_mat.Set(cl, col, from_mat.Get(cl, col)
                        / counts.Get(0, cl));
            else
                to_mat.Set(cl, col, from_mat.Get(cl, col));
        }
    }
}

/**
  Used to generate the a stream of random numbers on every processor but
  allow for a parallel and serial impl to generate identical results.
    NOTE: This only works if the data is distributed to processors in the
  same fashion as El::VC, El::STAR
**/
template <typename T>
class mpi_random_generator {
    public:
        // End range (end_range) is inclusive i.e random numbers will be
        //      in the inclusive interval (begin_range, end_range)
        mpi_random_generator(const size_t begin_range,
                const size_t end_range, const El::Unsigned rank,
                const size_t nprocs, const size_t seed) {
            this->_nprocs = nprocs;
            this->_gen = std::default_random_engine(seed);
            this->_rank = rank;
            this->_dist = std::uniform_int_distribution<T>(begin_range, end_range);
            init();
        }

        void init() {
            for (size_t i = 0; i < _rank; i++)
                _dist(_gen);
        }

        T next() {
            T ret = _dist(_gen);
            for (size_t i = 0; i < _nprocs-1; i++)
                _dist(_gen);
            return ret;
        }

    private:
        std::uniform_int_distribution<T> _dist;
        std::default_random_engine _gen;
        size_t _nprocs;
        El::Unsigned _rank;
};

/* arg0 and arg1 are data buffers stored in column major format
    NOTE: Note the access pattern of arg0 and arg1 are cache inefficient
*/
template <typename T>
T euclidean_distance(const T* arg0, const El::Int arg0_height,
        const T* arg1, const El::Int arg1_height, const El::Int width) {
    T dist = 0;
    size_t numprocd = 0;
    while (numprocd < width) {
        T _dist = arg0[numprocd*arg0_height] - arg1[numprocd*arg1_height];
        dist += (_dist*_dist);
        numprocd++;
    }

    return std::sqrt(dist);
}

template <typename T>
void kmeanspp_init(const El::DistMatrix<T, El::VC, El::STAR>& data,
        El::Matrix<T>& centroids, const El::Unsigned seed,
        const El::Unsigned k) {
    El::Matrix<T> local_data = data.LockedMatrix();
    // NOTE: Will fail if every proc doesn't have enough mem for this!
    El::Matrix<T> dist_v(1, data.Height());
    // Only localEntries should be set to max
    El::Fill(dist_v, std::numeric_limits<T>::max());

    // Choose c1 uniformly at random
    srand(seed);
     // This is a global index
    El::Unsigned gl_selected_rid = random() % data.Height();

    if (data.IsLocalRow(gl_selected_rid)) {
        El::Unsigned local_selected_rid = data.LocalRow(gl_selected_rid);
#if KM_DEBUG
        El::Output("Proc: ", rank, " assigning global r:",
                gl_selected_rid, ", local r: ", local_selected_rid,
                " as centroid: 0");
#endif
        // Add row to local clusters
        centroids(El::IR(0,1), El::IR(0, data.Width())) +=
            local_data(El::IR(local_selected_rid, (local_selected_rid+1)),
                    El::IR(0, data.Width()));
        dist_v.Set(0, gl_selected_rid, 0);
    }

    // Globally sync across processes
    El::AllReduce(centroids, El::mpi::COMM_WORLD, El::mpi::SUM);

    unsigned clust_idx = 0; // The number of clusters assigned

    // Choose next center c_i with weighted prob
    while ((clust_idx + 1) < k) {
        T cuml_dist = 0;
        for (size_t row = 0; row < local_data.Height(); row++) {
            // Do a distance step
            T dist = euclidean_distance<T>(local_data.LockedBuffer(row, 0),
                    local_data.Height(),
                    centroids.LockedBuffer(clust_idx, 0), centroids.Height(),
                    centroids.Width());
            if (dist < dist_v.Get(0, data.GlobalRow(row)))
                dist_v.Set(0, data.GlobalRow(row), dist);
            cuml_dist += dist_v.Get(0, data.GlobalRow(row));

        }

        El::AllReduce(dist_v, El::mpi::COMM_WORLD, El::mpi::MIN);
        T recv_cuml_dist = 0;
        El::mpi::AllReduce(&cuml_dist, &recv_cuml_dist, 1,
                El::mpi::SUM, El::mpi::COMM_WORLD);
        cuml_dist = recv_cuml_dist;
        assert(cuml_dist > 0);

        cuml_dist = (cuml_dist * ((double)random())) / (RAND_MAX - 1.0);
        clust_idx++;

        for (size_t row = 0; row < data.Height(); row++) {
            cuml_dist -= dist_v.Get(0, row);

            if (cuml_dist <= 0) {
                if (data.IsLocalRow(row)) {
#if KM_DEBUG
                    El::Output("Proc: ", rank, " assigning r: ", row,
                            " local r: ", data.LocalRow(row),
                            " as centroid: ", clust_idx);
#endif

                    centroids(El::IR(clust_idx, clust_idx+1), El::IR(0,
                                data.Width())) +=
                        local_data(El::IR(data.LocalRow(row),
                            (data.LocalRow(row)+1)), El::IR(0, data.Width()));
                }

                El::Broadcast(centroids, El::mpi::COMM_WORLD,
                        data.RowOwner(row));
                break;
            }
        }
        assert(cuml_dist <= 0);
    }
}

template <typename T>
void init_centroids(El::Matrix<T>& centroids, const El::DistMatrix<T, El::VC,
        El::STAR>& data, init_t init, const El::Unsigned k,
        const size_t nrow, const size_t ncol, const El::Unsigned seed,
        std::vector<El::Unsigned>& centroid_assignment,
        El::Matrix<El::Int>& assignment_count, const El::Unsigned rank) {
    El::Unsigned nprocs = El::mpi::Size(El::mpi::COMM_WORLD);

    switch(init) {
        case init_t::RANDOM: {
            // Get the local data first
            El::Matrix<double> local_data = data.LockedMatrix();
            mpi_random_generator<El::Unsigned> gen(0, k-1, rank, nprocs, seed);

            for (El::Unsigned row = 0; row < local_data.Height(); row++) {
                El::Unsigned chosen_centroid_id = gen.next();

#if KM_DEBUG
                El::Output("Row: ", data.GlobalRow(row),
                        " chose c: ", chosen_centroid_id);
#endif

                // Add row to local clusters
                centroids(El::IR(chosen_centroid_id,
                            chosen_centroid_id+1), El::IR(0, ncol)) +=
                    local_data(El::IR(row, (row+1)), El::IR(0, ncol));

                // Increase cluster count
                assignment_count.Set(0, chosen_centroid_id,
                        assignment_count.Get(0, chosen_centroid_id) + 1);

                // Note the rows membership
                centroid_assignment[row] = chosen_centroid_id;
            }

            // Now we must merge per proc centroids
            El::AllReduce(centroids, El::mpi::COMM_WORLD, El::mpi::SUM);
            El::AllReduce(assignment_count, El::mpi::COMM_WORLD, El::mpi::SUM);

            // Get the means of the global centroids
            row_mean(centroids, centroids, assignment_count);
            // Reset the assignment count
            El::Zero(assignment_count);
        }
        break;
        case init_t::FORGY: {
            mpi_random_generator<El::Unsigned>
                gen(0, nrow-1, rank, 1, seed);

            for (El::Unsigned cl = 0; cl < k; cl++) {
                El::Unsigned chosen = gen.next();
#if KM_DEBUG
                if (rank == root)
                    printf("Selecting point %u as centroid\n", chosen);
#endif
                for (El::Unsigned col = 0; col < ncol; col++)
                    centroids.Set(cl, col, data.Get(chosen, col));
            }
        }
        break;
        case init_t::NONE:
            // Do Nothing
            break;
        case init_t::PLUSPLUS:
            kmeanspp_init(data, centroids, seed, k);
            break;
        case init_t::BARBAR:
            throw std::runtime_error("Not yet implemented");
            break;
        case init_t::SKETCH:
            throw std::runtime_error("Not yet implemented");
            break;
        default:
            throw std::runtime_error("Unknown"
                    " intialization method!");
    }
}

init_t get_init_type(std::string init) {
    if (std::string("random") == init)
        return init_t::RANDOM;
    else if (std::string("forgy") == init)
        return init_t::FORGY;
    else if (std::string("plusplus") == init)
        return init_t::PLUSPLUS;
    else if (std::string("barbar") == init)
        return init_t::BARBAR;
    else if (std::string("sketch") == init)
        return init_t::SKETCH;
    else if (std::string("none") == init)
        return init_t::NONE;
    else {
        std::string err = std::string("Unknown "
                "intialization method '") + init + "'";
        throw std::runtime_error(err);
    }
}

// Get the sum of a matrix
template <typename T>
T sum(const El::Matrix<T>& mat) {
    T total = 0;
    for (size_t row = 0; row < mat.Height(); row++)
        for (size_t col = 0; col < mat.Width(); col++)
            total += mat.Get(row, col);
    return total;
}

template <typename T>
void naive_kmeans(const El::Matrix<T>& data, El::Matrix<T>& centroids,
        El::Matrix<T>& local_centroids,
        El::Matrix<El::Int>& assignment_count,
        std::vector<El::Unsigned>& centroid_assignment,
        El::Unsigned& nchanged) {
    // Populate per process centroids and keep track of how many
    const El::Unsigned k = centroids.Height();
    const size_t ncol =  centroids.Width();
    const size_t nrow = data.Height();

#if KM_DEBUG
    if (rank == 0) {
        El::Output("Process 0 has ", nrow, " rows");
    }
#endif

    for (size_t row = 0; row < nrow; row++) {
        El::Unsigned assigned_centroid_id = INVALID_ID;
        double dist = std::numeric_limits<double>::max();
        double best = std::numeric_limits<double>::max();

        for (El::Unsigned cl = 0; cl < k; cl++) {
            dist = euclidean_distance<T>(data.LockedBuffer(row, 0),
                    data.Height(), centroids.LockedBuffer(cl, 0),
                    centroids.Height(), ncol);
            if (dist < best) {
                best = dist;
                assigned_centroid_id = cl;
            }
        }

        assert(assigned_centroid_id != INVALID_ID);

        // Have I changed clusters ?
        if (centroid_assignment[row] != assigned_centroid_id) {
#if KM_DEBUG
            El::Output("Row: ", row, " => OC: ", centroid_assignment[row],
                    " NC: ", assigned_centroid_id, "\n");
#endif
            centroid_assignment[row] = assigned_centroid_id;
            nchanged++;
        }

        // Add row to local clusters
        local_centroids(El::IR(assigned_centroid_id,
                    assigned_centroid_id+1), El::IR(0, ncol)) +=
            data(El::IR(row, (row+1)), El::IR(0, ncol));

        // Increase cluster count
        assignment_count.Set(0, assigned_centroid_id,
                assignment_count.Get(0, assigned_centroid_id) + 1);
    }

    assert(sum(assignment_count) == data.Height());
}

void get_global_assignments(const std::vector<El::Unsigned>&
        centroid_assignment, std::vector<El::Unsigned>&
        gl_centroid_assignments, const size_t local_height) {
    El::mpi::Comm comm = El::mpi::COMM_WORLD;
    El::Unsigned nprocs = El::mpi::Size(comm);
    El::Unsigned rank = El::mpi::Rank(comm);

    // We could compute how many rows each proc has, but this is simpler
    El::Matrix<El::Int> rpp;
    El::Zeros(rpp, 1, nprocs);
    rpp.Set(0, rank, local_height);
    El::AllReduce(rpp, comm, El::mpi::SUM);

    std::vector<std::vector<El::Unsigned> > all_centroid_assignment(nprocs);
    if (rank != root) {
        El::mpi::Send(&centroid_assignment[0], (int)centroid_assignment.size(),
                root, comm);
    }
    else {
        for (El::Unsigned p = 0; p < nprocs; p++)
            all_centroid_assignment[p].resize(rpp.Get(0, p));

        // Copy the root for ease of accumulation
        all_centroid_assignment[root] = centroid_assignment;
        for (El::Unsigned srank = 1; srank < nprocs; srank++)
            El::mpi::Recv(&((all_centroid_assignment[srank])[0]),
                    rpp.Get(0, srank), srank, comm);

        // Cache UNfriendly access pattern here
        for (size_t memb = 0; memb <
                all_centroid_assignment[0].size(); memb++) {
            for (El::Unsigned p = 0; p < nprocs; p++) {
                // Some may have fewer
                if (memb < all_centroid_assignment[p].size())
                    gl_centroid_assignments.
                        push_back(all_centroid_assignment[p][memb]);
            }
        }
    }
}
}

namespace skylark { namespace ml {
    /**
      * Type used to return items from the computation of
      *     kmeans.
      */
class kmeans_t {
    public:
        std::vector<El::Unsigned> gl_centroid_assignments;
        std::vector<El::Int> assignment_count;
        El::Unsigned iters;

        kmeans_t(std::vector<El::Unsigned>& gl_centroid_assignments,
                El::Int* assignment_count_buf, const size_t k,
                const El::Unsigned iters) {
            this->gl_centroid_assignments = gl_centroid_assignments;
            this->iters = iters;
            this->assignment_count.resize(k);
            std::copy(assignment_count_buf, assignment_count_buf + k,
                    assignment_count.begin());
        }
};

/**
  * Driver function for kmeans.
  */
kmeans_t run_kmeans(El::DistMatrix<double, El::VC, El::STAR>& data,
        El::Matrix<double>& centroids, const El::Unsigned k,
        const size_t ncol, const double tol, const std::string init,
        const El::Int seed, const El::Unsigned max_iters,
        const El::Unsigned rank) {
    El::Unsigned nchanged = 0;

    // Count # points in a each centroid per proc
    El::Matrix<El::Int> assignment_count(1, k);
    El::Zero(assignment_count);

    El::Matrix<double> local_centroids(k, ncol);
    El::Zero(local_centroids);

    std::vector<El::Unsigned> centroid_assignment;
    centroid_assignment.assign(data.LocalHeight(), INVALID_ID);

    init_centroids<double>(centroids, data, get_init_type(init),
            k, data.Height(), ncol, seed, centroid_assignment,
            assignment_count, rank);

    // Run iterations
    double perc_changed = std::numeric_limits<double>::max();
    El::Unsigned iters = 0;
    bool converged = false;

    El::mpi::Comm comm = El::mpi::COMM_WORLD;

    while (perc_changed > tol && iters < max_iters) {
        if (rank == root)
            El::Output("Running  iteration ", iters, " ...\n");

        naive_kmeans<double>(data.LockedMatrix(), centroids, local_centroids,
                assignment_count, centroid_assignment, nchanged);
        iters++;

        El::Unsigned recv_nchanged = INVALID_ID;
        El::mpi::AllReduce(&nchanged, &recv_nchanged, 1, El::mpi::SUM, comm);
        assert(recv_nchanged != INVALID_ID);
        nchanged = recv_nchanged;

        El::AllReduce(assignment_count, comm, El::mpi::SUM);

        if (rank == root)
            El::Output("Global nchanged: ", nchanged);

        perc_changed = (double)nchanged/data.Height(); //Global perc change
        if (perc_changed <= tol) {
            converged = true;
            if (rank == root) {
                El::Output("Algorithm converged in ", iters,
                        " iterations!");
                El::Print(assignment_count, "\nFinal assingment count");
            }
            break;
        }

#if KM_DEBUG
        if (rank == root) El::Output("Reducing local centroids ...\n");
#endif

        // Aggregate all local centroids
        El::AllReduce(local_centroids, comm, El::mpi::SUM);
        // Get the means of the local centroids
        row_mean(local_centroids, centroids, assignment_count);

#if KM_DEBUG
        if (rank == root)
            El::Print(centroids, "Updated centroids for root");
#endif

        // Reset
        nchanged = 0;
        El::Zero(local_centroids);
        El::Zero(assignment_count);
    }

    // Get the centroid assignments to the root
    std::vector<El::Unsigned> gl_centroid_assignments;
    get_global_assignments(centroid_assignment,
            gl_centroid_assignments, data.LocalHeight());

#if 1
    if (rank == root) {
        El::Output("Centroid assignment:");
        skyutil::PrettyPrinter<El::Unsigned>::print(gl_centroid_assignments);
    }

    if (!converged && rank == root)
        El::Output("Algorithm failed to converge in ",
                iters, " iterations\n");
#endif

    return kmeans_t(gl_centroid_assignments,
            assignment_count.Buffer(), k, iters);
}

} } // namespace skylark::ml
#endif /* SKYLARK_KMEANS_HPP */
