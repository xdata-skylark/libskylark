# Random123
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
find_package(Random123 REQUIRED)
if (Random123_FOUND)
  include_directories (${Random123_INCLUDE_DIR})
endif (Random123_FOUND)
