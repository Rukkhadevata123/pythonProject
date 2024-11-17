cmake_minimum_required(VERSION 3.5)

project(KvantumTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_PREFIX_PATH "/usr/share/qt6")  # 修改为你的 Qt6 安装路径

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(KvantumTest main.cpp)

target_link_libraries(KvantumTest Qt6::Widgets)