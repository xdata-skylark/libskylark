option (BUILD_NLA "Whether we should build the numerical linear algebra solves" ON)
if (BUILD_NLA)
  add_subdirectory(cli/nla)
endif (BUILD_NLA)
