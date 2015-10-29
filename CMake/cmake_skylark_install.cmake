#----------------------------------------------------------------------------
# Write out the configure file
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
                ${CMAKE_BINARY_DIR}/config.h)
install (FILES ${CMAKE_BINARY_DIR}/config.h DESTINATION include/skylark/)

#----------------------------------------------------------------------------
# Install skylark
#TODO: global recurse is evil (cmake install does not get notified when new
#      files are added). We should compile the header list differently (e.g.
#      install an individual CMakeList files in directories).
file (GLOB_RECURSE HEADER_LIST RELATIVE ${CMAKE_SOURCE_DIR} *.hpp )
foreach (HEADER ${HEADER_LIST})
  string (REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
  install (FILES ${HEADER} DESTINATION include/skylark/${DIR})
endforeach (HEADER)

#----------------------------------------------------------------------------
# Uninstall skylark
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/CMake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
