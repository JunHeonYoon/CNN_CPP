cmake_minimum_required(VERSION 3.10)

# 프로젝트 이름 설정
project(CNN_CPP)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-DBUILD_DIRECTORY=\"${CMAKE_SOURCE_DIR}\")

# Eigen 라이브러리 찾기
find_package(Eigen3 REQUIRED)

# include 디렉토리 추가
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# 소스 파일 목록 설정
set(SOURCES
    src/BatchNorm1d.cpp
    src/Conv1d.cpp
    src/Linear.cpp
    src/test.cpp
)

# 실행 파일 생성
add_executable(test ${SOURCES})

# Eigen3 라이브러리 링크
target_link_libraries(test Eigen3::Eigen)

# 빌드 출력 디렉토리 설정
set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)
