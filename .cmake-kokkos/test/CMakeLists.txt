set(all_execs
  test_atadd.kokkos
  test_random.kokkos
  test_cuda.kokkos
)

set(source_dir ${PROJECT_SOURCE_DIR}/../test)

foreach(source_file IN LISTS all_execs)
  get_filename_component(exec_name ${source_file} NAME_WE)
  set(renamed_source_file ${CMAKE_CURRENT_BINARY_DIR}/${exec_name}.cpp)
  configure_file("${source_dir}/${source_file}" ${renamed_source_file} COPYONLY)
  add_executable(${exec_name} ${renamed_source_file})
  target_link_libraries(${exec_name} PRIVATE Kokkos::kokkos)
endforeach()
