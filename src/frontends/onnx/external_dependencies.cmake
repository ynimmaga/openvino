# external_dependencies.cmake
# This file now ONLY fetches GSL, as OpenVINO provides the other dependencies.

include(FetchContent)

message(STATUS "Configuring GSL dependency for OpenVINO EP...")

FetchContent_Declare(gsl
    GIT_REPOSITORY https://github.com/microsoft/GSL.git
    GIT_TAG        v4.0.0
)

set(GSL_TEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(gsl)

message(STATUS "External dependencies configuration completed.")