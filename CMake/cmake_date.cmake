set (BUILD_DATE 0)
EXECUTE_PROCESS(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND date
    OUTPUT_VARIABLE DATE_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
)
set (BUILD_DATE \"${DATE_OUT}\")
