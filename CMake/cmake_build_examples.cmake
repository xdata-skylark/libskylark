option (BUILD_EXAMPLES "Whether we should build the examples" ON)
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif (BUILD_EXAMPLES)
