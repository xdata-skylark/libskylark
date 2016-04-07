option (BUILD_PYTHON "Whether we should build the python interface" ON)
if (BUILD_PYTHON)

  find_package(PythonInterp REQUIRED)
  if (DEFINED ENV{PYTHON_SITE_PACKAGES})
    set (PYTHON_SITE_PACKAGES
      $ENV{PYTHON_SITE_PACKAGES}
    )
    set (PYTHON_SYSTEM_WIDE_INSTALL 0)
  else (DEFINED ENV{PYTHON_SITE_PACKAGES})
    if(${PYTHON_VERSION_MAJOR} GREATER 2)
      message (FATAL_ERROR "Currently we only support Python 2.x (version ${PYTHON_VERSION_STRING} found). Please disable the BUILD_PYTHON option.")
    endif(${PYTHON_VERSION_MAJOR} GREATER 2)

    execute_process (
      COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set (PYTHON_SYSTEM_WIDE_INSTALL 1)
  endif (DEFINED ENV{PYTHON_SITE_PACKAGES})
  message (STATUS "Installing python modules in: ${PYTHON_SITE_PACKAGES}")

  add_subdirectory(python-skylark)
endif (BUILD_PYTHON)
