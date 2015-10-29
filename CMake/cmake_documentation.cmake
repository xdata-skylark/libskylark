#----------------------------------------------------------------------------
# Doxygen
find_package(Doxygen)
if (DOXYGEN_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY
    )
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif (DOXYGEN_FOUND)

#----------------------------------------------------------------------------
# Sphinx
find_package(Sphinx)
if (SPHINX_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/conf.py
        ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY
    )
    add_custom_target(sphinx-doc
        ${SPHINX_EXECUTABLE}
        -q -b html
        -d "${CMAKE_CURRENT_BINARY_DIR}/_cache"
        "${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx"
        "${CMAKE_CURRENT_BINARY_DIR}/Documentation/sphinx"
        COMMENT "Building HTML documentation with Sphinx"
    )
endif (SPHINX_FOUND)
