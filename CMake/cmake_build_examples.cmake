option (BUILD_EXAMPLES "Whether we should build the examples" OFF)
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif (BUILD_EXAMPLES)
