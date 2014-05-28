if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)
  add_executable(hilbert train.cpp)
  target_link_libraries(hilbert
                        ${SKYLARK_LIBS}
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES}
                        ${HDF5_LIBRARIES}
                        ${ZLIB_LIBRARIES})

endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

