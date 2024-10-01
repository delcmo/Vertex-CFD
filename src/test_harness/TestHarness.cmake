set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/src/test_harness)

##--------------------------------------------------------------------------##
## Test Macro
##--------------------------------------------------------------------------##
macro(VertexCFD_add_tests)
  set(options OPTIONAL MPI)
  set(oneValueArgs)
  set(multiValueArgs LIBS NAMES)
  cmake_parse_arguments(VERTEXCFD_UNIT_TEST "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN} )
  set(VERTEXCFD_UNIT_TEST_MPIEXEC_NUMPROCS 1)
  foreach( _procs 2 4 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_procs})
      list(APPEND VERTEXCFD_UNIT_TEST_MPIEXEC_NUMPROCS ${_procs})
    endif()
  endforeach()
  set(VERTEXCFD_UNIT_TEST_NUMTHREADS 1)
  foreach( _threads 2 4 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_threads})
      list(APPEND VERTEXCFD_UNIT_TEST_NUMTHREADS ${_threads})
    endif()
  endforeach()
  if(MPIEXEC_MAX_NUMPROCS GREATER 4)
    list(APPEND VERTEXCFD_UNIT_TEST_MPIEXEC_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
    list(APPEND VERTEXCFD_UNIT_TEST_NUMTHREADS ${MPIEXEC_MAX_NUMPROCS})
  endif()
  set(VERTEXCFD_UNIT_TEST_MAIN ${TEST_HARNESS_DIR}/test_main.cpp)
  set(_device ${VERTEXCFD_KOKKOS_DEVICE_TYPE})
  set(_dir ${CMAKE_CURRENT_SOURCE_DIR})
  foreach(_test ${VERTEXCFD_UNIT_TEST_NAMES})
    set(_file ${_dir}/tst${_test}.cpp)
    set(_target VertexCFD_${_test}_test_${_device})
    add_executable(${_target} ${_file} ${VERTEXCFD_UNIT_TEST_MAIN})
    target_include_directories(${_target} PRIVATE ${_dir}
      ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_HARNESS_DIR})
    target_link_libraries(${_target} PRIVATE ${VERTEXCFD_UNIT_TEST_LIBS} GTest::gtest Kokkos::kokkos MPI::MPI_CXX )
    if(VERTEXCFD_UNIT_TEST_MPI)
      foreach(_procs ${VERTEXCFD_UNIT_TEST_MPIEXEC_NUMPROCS})
        # NOTE: When moving to CMake 3.10+ make sure to use MPIEXEC_EXECUTABLE instead
        if(_device STREQUAL PTHREAD OR _device STREQUAL OPENMP)
          foreach(_threads ${VERTEXCFD_UNIT_TEST_NUMTHREADS})
            math(EXPR _total_threads "${_procs} * ${_threads}")
            if(_total_threads GREATER MPIEXEC_MAX_NUMPROCS)
              break()
            endif()
            set(_test_name ${_target}_np_${_procs}_nt_${_threads})
            add_test(NAME ${_test_name} COMMAND
              ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_procs} ${MPIEXEC_PREFLAGS}
              ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args} --kokkos-threads=${_threads})
            set_property(TEST ${_test_name} PROPERTY PROCESSORS ${_total_threads})
            set_property(TEST ${_test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${_threads}")
          endforeach()
        else()
          set(_test_name ${_target}_np_${_procs})
          add_test(NAME ${_target}_np_${_procs} COMMAND
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_procs} ${MPIEXEC_PREFLAGS}
            ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args} --kokkos-threads=1)
          set_property(TEST ${_test_name} PROPERTY PROCESSORS ${_procs})
        endif()
      endforeach()
    else()
      if(_device STREQUAL OPENMP)
        foreach(_threads ${VERTEXCFD_UNIT_TEST_NUMTHREADS})
          set(_test_name ${_target}_nt_${_threads})
          add_test(NAME ${_target}_nt_${_threads} COMMAND
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
            ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args} --kokkos-threads=${_threads})
          set_property(TEST ${_test_name} PROPERTY PROCESSORS ${_threads})
          set_property(TEST ${_test_name} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${_threads}")
        endforeach()
      else()
        add_test(NAME ${_target} COMMAND
          ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
          ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args} --kokkos-threads=1)
        set_property(TEST ${_target} PROPERTY PROCESSORS 1)
      endif()
    endif()
  endforeach()
endmacro()
