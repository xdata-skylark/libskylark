/*
 * timer.hpp
 *
 *  Created on: Feb 17, 2014
 *      Author: vikas
 */

#ifndef SKYLARK_PROFILER_HPP
#define SKYLARK_PROFILER_HPP

#ifdef SKYLARK_HAVE_PROFILER

#define SKYLARK_TIMER_REPORT(X) \
    if (context.rank == 0) { \
            double X##_time_min, X##_time_max, X##_time_ave; \
            boost::mpi::reduce(context.comm, \
                X##_time, \
                X##_time_min, \
                boost::mpi::minimum<double>(), \
                0); \
            boost::mpi::reduce(context.comm, \
                X##_time, \
                X##_time_max, \
                boost::mpi::maximum<double>(), \
                0); \
            boost::mpi::reduce(context.comm, \
                X##_time, \
                X##_time_ave, \
                std::plus<double>(), \
                0); \
            X##_time_ave = X##_time_ave / context.size; \
            std::cout << #X << " time (secs)" << std::endl; \
            std::cout << "Min: " << X##_time_min << std::endl; \
            std::cout << "Max: " << X##_time_max << std::endl; \
            std::cout << "Ave: " << X##_time_ave << std::endl; \
            std::cout << std::endl; \
        } else { \
            boost::mpi::reduce(context.comm, \
                X##_time, \
                boost::mpi::minimum<double>(), \
                0); \
            boost::mpi::reduce(context.comm, \
                X##_time, \
                boost::mpi::maximum<double>(), \
                0); \
            boost::mpi::reduce(context.comm, \
                X##_time, \
                std::plus<double>(), \
                0); \
          } \


#endif

#endif // SKYLARK_PROFILER_HPP
