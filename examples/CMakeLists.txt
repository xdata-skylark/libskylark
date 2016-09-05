



add_executable(kernel_regression kernel_regression.cpp)
target_link_libraries(kernel_regression
  ${Elemental_LIBRARY}
  ${OPTIONAL_LIBS}
  ${Pmrrr_LIBRARY}
  ${Metis_LIBRARY}
  ${SKYLARK_LIBS}
  ${Boost_LIBRARIES})

install_targets(/bin/skylark_examples kernel_regression)

