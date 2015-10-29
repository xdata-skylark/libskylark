option (USE_PROFILER "Use Skylark profiling tools" OFF)
if (USE_PROFILER)
  set (SKYLARK_HAVE_PROFILER
       1
       CACHE
       STRING
       "Enables use of profiling extensions"
       FORCE)
endif (USE_PROFILER)

