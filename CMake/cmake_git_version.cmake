# we start by gathering some revision version information, first we try GIT
execute_process (
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse HEAD
    RESULT_VARIABLE GIT_REPO
    OUTPUT_VARIABLE GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (GIT_REPO EQUAL 0)
  message (STATUS "Building git version ${GIT_REV}")
  set (GIT_REVISION \"${GIT_REV}\")
else (GIT_REPO EQUAL 0)
  message (STATUS "No git repository found.")
  set (GIT_REVISION 0)
endif (GIT_REPO EQUAL 0)
