option (BUILD_ML "Whether we should build the machine learning solvers" ON)
if (BUILD_ML)
  add_subdirectory(ml)
endif (BUILD_ML)
