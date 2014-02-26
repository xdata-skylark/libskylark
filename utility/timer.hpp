#ifndef SKYLARK_TIMER_HPP
#define SKYLARK_TIMER_HPP

#ifdef SKYLARK_HAVE_PROFILER

#define SKYLARK_TIMER_INITIALIZE(X) \
    boost::mpi::timer X##_timer; \
    double X##_time = 0.0; \

#define SKYLARK_TIMER_RESTART(X) \
    X##_timer.restart(); \

#define SKYLARK_TIMER_ACCUMULATE(X) \
    X##_time += X##_timer.elapsed(); \

#define SKYLARK_TIMER_PRINT(X) \
    if (base_data_t::context.rank == 0) { \
            double X##_time_min, X##_time_max, X##_time_ave; \
            boost::mpi::reduce(base_data_t::context.comm, \
                X##_time, \
                X##_time_min, \
                boost::mpi::minimum<double>(), \
                0); \
            boost::mpi::reduce(base_data_t::context.comm, \
                X##_time, \
                X##_time_max, \
                boost::mpi::maximum<double>(), \
                0); \
            boost::mpi::reduce(base_data_t::context.comm, \
                X##_time, \
                X##_time_ave, \
                std::plus<double>(), \
                0); \
            X##_time_ave = X##_time_ave / base_data_t::context.size; \
            std::cout << #X << " time (secs)" << std::endl; \
            std::cout << "Min: " << X##_time_min << std::endl; \
            std::cout << "Max: " << X##_time_max << std::endl; \
            std::cout << "Ave: " << X##_time_ave << std::endl; \
            std::cout << std::endl; \
        } else { \
            boost::mpi::reduce(base_data_t::context.comm, \
                X##_time, \
                boost::mpi::minimum<double>(), \
                0); \
            boost::mpi::reduce(base_data_t::context.comm, \
                X##_time, \
                boost::mpi::maximum<double>(), \
                0); \
            boost::mpi::reduce(base_data_t::context.comm, \
                X##_time, \
                std::plus<double>(), \
                0); \
          } \

#endif //SKYLARK_HAVE_PROFILER

#endif //SKYLARK_TIMER_HPP
