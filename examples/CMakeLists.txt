if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

  add_executable(hp_dense hp_dense.cpp)
  target_link_libraries(hp_dense
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})

  add_executable(elemental elemental.cpp)
  target_link_libraries(elemental
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})

  add_executable(regression regression.cpp)
  target_link_libraries(regression
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})

  add_executable(least_squares least_squares)
  target_link_libraries(least_squares
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})


  add_executable(rand_svd rand_svd.cpp)
  target_link_libraries(rand_svd
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})

if (FALSE)
# Commenting this out for now as we don't have a functioning
# SVD implementation that uses sketching
  add_executable(svd svd.cpp)
  target_link_libraries(svd
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
endif(FALSE)

endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

if(FALSE)
if (SKYLARK_HAVE_COMBBLAS AND SKYLARK_HAVE_ELEMENTAL)
  add_executable(lsqr lsqr.cpp)
  target_link_libraries(lsqr
                        ${Elemental_LIBRARY}
                        ${CombBLAS_LIBRARIES}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
endif (SKYLARK_HAVE_COMBBLAS AND SKYLARK_HAVE_ELEMENTAL)
endif(FALSE)
