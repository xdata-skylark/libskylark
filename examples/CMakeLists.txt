if (SKYLARK_HAVE_FFTW)

  add_executable(hp_dense hp_dense.cpp)
  target_link_libraries(hp_dense
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  install_targets(/bin/examples hp_dense)

  add_executable(elemental elemental.cpp)
  target_link_libraries(elemental
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  install_targets(/bin/examples elemental)

  add_executable(regression regression.cpp)
  target_link_libraries(regression
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  install_targets(/bin/examples regression)

  add_executable(least_squares least_squares)
  target_link_libraries(least_squares
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  install_targets(/bin/examples least_squares)

  add_executable(community community)
  target_link_libraries(community
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES}
                        ${HDF5_LIBRARIES}
                        ${ZLIB_LIBRARIES})
  install_targets(/bin/examples community)

  add_executable(rand_svd rand_svd.cpp)
  target_link_libraries(rand_svd
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  install_targets(/bin/examples rand_svd)

endif (SKYLARK_HAVE_FFTW)

if (SKYLARK_HAVE_HDF5)
  add_executable(condest condest.cpp)
  target_link_libraries(condest
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES}
                        ${HDF5_LIBRARIES}
                        ${ZLIB_LIBRARIES})
  install_targets(/bin/examples condest)
endif (SKYLARK_HAVE_HDF5)

if (SKYLARK_HAVE_OPENMP AND SKYLARK_HAVE_HDF5)
  add_executable(asynch asynch.cpp)
  target_link_libraries(asynch
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES}
                        ${HDF5_LIBRARIES}
                        ${ZLIB_LIBRARIES})
  install_targets(/bin/examples asynch)
endif (SKYLARK_HAVE_OPENMP AND SKYLARK_HAVE_HDF5)
