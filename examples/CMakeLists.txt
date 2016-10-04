add_executable(hp_dense hp_dense.cpp)
target_link_libraries(hp_dense
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})
install_targets(/bin/skylark_examples hp_dense)

add_executable(elemental elemental.cpp)
target_link_libraries(elemental
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})
install_targets(/bin/skylark_examples elemental)

add_executable(regression regression.cpp)
target_link_libraries(regression
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})
install_targets(/bin/skylark_examples regression)

add_executable(random_features random_features.cpp)
target_link_libraries(random_features
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})
install_targets(/bin/skylark_examples random_features)

add_executable(least_squares least_squares)
target_link_libraries(least_squares
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})
install_targets(/bin/skylark_examples least_squares)

add_executable(kernel_regression kernel_regression.cpp)
target_link_libraries(kernel_regression
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin/skylark_examples kernel_regression)

add_executable(kmeans kmeans.cpp)
target_link_libraries(kmeans
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})
install_targets(/bin/skylark_examples kmeans)

if (SKYLARK_HAVE_HDF5)
  add_executable(condest condest.cpp)
  target_link_libraries(condest
    ${Elemental_LIBRARY}
    ${OPTIONAL_LIBS}
    ${Pmrrr_LIBRARY}
    ${Metis_LIBRARY}
    ${SKYLARK_LIBS}
    ${Boost_LIBRARIES})
  install_targets(/bin/skylark_examples condest)
endif (SKYLARK_HAVE_HDF5)

if (SKYLARK_HAVE_OPENMP AND SKYLARK_HAVE_HDF5)
  add_executable(asynch asynch.cpp)
  target_link_libraries(asynch
    ${Elemental_LIBRARY}
    ${OPTIONAL_LIBS}
    ${Pmrrr_LIBRARY}
    ${Metis_LIBRARY}
    ${SKYLARK_LIBS}
    ${Boost_LIBRARIES})
  install_targets(/bin/skylark_examples asynch)
endif (SKYLARK_HAVE_OPENMP AND SKYLARK_HAVE_HDF5)
