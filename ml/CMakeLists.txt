if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)
  add_executable(skylark_ml train.cpp)
  target_link_libraries(skylark_ml
                        ${SKYLARK_LIBS}
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES}
                        ${HDF5_LIBRARIES}
                        ${ZLIB_LIBRARIES})
   install_targets(/bin/ml skylark_ml)

endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

