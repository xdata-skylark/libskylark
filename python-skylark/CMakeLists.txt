find_program(PYTHON "python")

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/skylark/*.py")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command(OUTPUT ${OUTPUT}/python-skylark
                       COMMAND ${PYTHON} ${SETUP_PY} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/python-skylark
                       DEPENDS ${DEPS})

    add_custom_target(python-skylark ALL DEPENDS ${OUTPUT}/python-skylark)

    if(PYTHON_SYSTEM_WIDE_INSTALL)
        install(
            CODE
            "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)"
        )
    else(PYTHON_SYSTEM_WIDE_INSTALL)
        install(
            CODE
            "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${PYTHON_SITE_PACKAGES})"
        )
    endif(PYTHON_SYSTEM_WIDE_INSTALL)
endif (PYTHON)
