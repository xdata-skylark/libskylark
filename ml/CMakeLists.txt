add_executable(skylark_ml skylark_ml.cpp)

target_link_libraries(skylark_ml
  ${SKYLARK_LIBS}
  ${Elemental_LIBRARY}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${OPTIONAL_LIBS}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${ZLIB_LIBRARIES})

install_targets(/bin/ml skylark_ml)



