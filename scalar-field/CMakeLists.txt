cmake_minimum_required(VERSION 3.0)
project(ScalarField CXX)


# find_package(Threads REQUIRED)


file(GLOB Main_Code_Pile
	"*.cpp")

add_executable(scalar_field ${Main_Code_Pile})

set_target_properties(scalar_field PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# target_link_libraries(scalar_field Threads::Threads)
