option (BUILD_CAPI "Whether we should build the C bindings" ON)
if (BUILD_CAPI)
  add_subdirectory(capi)
endif (BUILD_CAPI)
