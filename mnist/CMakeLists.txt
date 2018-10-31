cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(mnist CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_STANDARD 17)

file(GLOB HRDS
    include/mnist/*.h
)
file(GLOB SRCS
    src/*.cpp
)

add_library(${PROJECT_NAME}
    ${SRCS}
    ${HRDS}
)

target_link_libraries(${PROJECT_NAME}
    ${CONAN_LIBS_CATBOOST}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    include
    3rd_party/tensorflow/include
)

set(DEMO_COMMON_HDRS
    demo/helpers.h
)

set(DEMO_COMMON_SRCS
    demo/helpers.cpp
)

add_executable(tf_sample
    demo/tf_sample.cpp
    ${DEMO_COMMON_SRCS}
    ${DEMO_COMMON_HDRS}
)

target_include_directories(tf_sample PRIVATE
    3rd_party/tensorflow/include
)

# link_directories(
#     ${CMAKE_SOURCE_DIR}/3rd_party/tensorflow/lib
# )

target_link_libraries(tf_sample
    ${PROJECT_NAME}
    ${CONAN_LIBS}
    ${CMAKE_SOURCE_DIR}/3rd_party/tensorflow/lib/libtensorflow.so
)

# add_executable(demo_logreg
#     demo/demo_logreg.cpp
#     ${DEMO_COMMON_SRCS}
#     ${DEMO_COMMON_HDRS}
# )

# target_include_directories(demo_logreg PUBLIC
#     include
# )

# target_link_libraries(demo_logreg
#     ${PROJECT_NAME}
# )

# add_executable(demo_linreg
#     demo/demo_linreg.cpp
#     ${DEMO_COMMON_SRCS}
#     ${DEMO_COMMON_HDRS}
# )

# target_include_directories(demo_linreg PUBLIC
#     include
# )

# target_link_libraries(demo_linreg
#     ${PROJECT_NAME}
# )


enable_testing()

file(GLOB TEST_SRCS
    tests/*.cpp
)

add_executable(test_${PROJECT_NAME}
    ${SRCS}
    ${HRDS}
    ${TEST_SRCS}
    ${DEMO_COMMON_SRCS}
    ${DEMO_COMMON_HDRS}
)

target_include_directories(test_${PROJECT_NAME} PUBLIC
    include
    demo
    3rd_party/tensorflow/include
)

target_compile_options(test_${PROJECT_NAME}
    PRIVATE -fsanitize=address
            -fsanitize=undefined
            -g
            -fomit-frame-pointer
)

target_link_libraries(test_${PROJECT_NAME}
     -fsanitize=address
    -fsanitize=undefined
    ${CONAN_LIBS_GTEST}
    ${CMAKE_SOURCE_DIR}/3rd_party/tensorflow/lib/libtensorflow.so
)