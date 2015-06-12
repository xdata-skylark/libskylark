add_executable(skylark_ml skylark_ml.cpp)

target_link_libraries(skylark_ml
  ${SKYLARK_LIBS}
  ${Elemental_LIBRARY}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${OPTIONAL_LIBS}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin skylark_ml)

add_executable(skylark_community skylark_community.cpp)

target_link_libraries(skylark_community
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})
install_targets(/bin skylark_community)


if (SKYLARK_HAVE_HDF5)
add_executable(skylark_convert2hdf5 skylark_convert2hdf5.cpp)

target_link_libraries(skylark_convert2hdf5
  ${SKYLARK_LIBS}
  ${Elemental_LIBRARY}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${OPTIONAL_LIBS}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})
endif (SKYLARK_HAVE_HDF5)

