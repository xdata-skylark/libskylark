# Find Boost with the relevant packages --- Use dynamic boost!
# Without dynamic linking, it's tough to create python bindings.
set (BOOST_ROOT $ENV{BOOST_ROOT})
# turn off system paths if BOOST_ROOT is defined
if (BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS ON)
  set(Boost_NO_BOOST_CMAKE ON)
endif (BOOST_ROOT)

set(BOOST_MIN_VERSION 1.53.0)
find_package (Boost COMPONENTS filesystem)
find_package (Boost REQUIRED mpi program_options serialization system)
if (Boost_FOUND)
  set (SKYLARK_HAVE_BOOST
       1
       CACHE
       STRING
       "Enables use of Boost Libraries"
       FORCE)
  include_directories (${Boost_INCLUDE_DIRS})
  link_directories (${Boost_LIBRARY_DIRS})
  message(STATUS "Found Boost: ${Boost_INCLUDE_DIRS}" )
endif (Boost_FOUND)

if (Boost_FILESYSTEM_FOUND)
  set (SKYLARK_HAVE_BOOST_FILESYSTEM
       1
       CACHE
       STRING
       "Enables use of Boost Libraries"
       FORCE)
  message(STATUS "Found Boost Filesystem: ${Boost_INCLUDE_DIRS}" )
endif (Boost_FILESYSTEM_FOUND)
