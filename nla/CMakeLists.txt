add_executable(skylark_svd skylark_svd.cpp)

target_link_libraries(skylark_svd
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin skylark_svd)

add_executable(massive_skylark_svd massive_skylark_svd.cpp)

target_link_libraries(massive_skylark_svd
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin massive_skylark_svd)

add_executable(massive_skylark_least_squares massive_skylark_least_squares.cpp)

target_link_libraries(massive_skylark_least_squares
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin massive_skylark_least_squares)

add_executable(skylark_linear skylark_linear.cpp)

target_link_libraries(skylark_linear
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin skylark_linear)