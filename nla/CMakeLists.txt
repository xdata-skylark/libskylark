add_executable(skylark_svd skylark_svd.cpp)

target_link_libraries(skylark_svd
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin skylark_svd)

add_executable(skylark_linear skylark_linear.cpp)

target_link_libraries(skylark_linear
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin skylark_linear)