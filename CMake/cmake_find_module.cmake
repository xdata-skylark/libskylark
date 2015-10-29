#----------------------------------------------------------------------------
# configure file
#FIXME: is there a cleaner way to achieve this?
get_directory_property( DirDefs COMPILE_DEFINITIONS )
foreach( d ${DirDefs} )
    set (SKYLARK_DEFS
        ${SKYLARK_DEFS}
        "-D${d}")
endforeach()

configure_file (
  ${CMAKE_CURRENT_SOURCE_DIR}/CMake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake
)

install (
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME}"
  RENAME ${PROJECT_NAME}Config.cmake
)
