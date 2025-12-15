cmake_minimum_required(VERSION 3.13)

project(mnn_tts_SDK VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_BERTVITS2 "Build BertVit2 TTS " OFF)
option(BUILD_PIPER "Build PIPER TTS " OFF)
option(BUILD_SUPERTONIC "Build Supertonic TTS " ON)

# 设置MNN路径
set(MNN_ROOT "/Users/ws/ws/projects/sdk/mnn_sdks/MNN_All/MNN-MacOS-CPU-GPU/Dynamic/MNN.framework")

# 包含目录
include_directories(
	${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/third_party/include
	${CMAKE_CURRENT_LIST_DIR}/third_party/include/nlohmann
    ${MNN_ROOT}/Headers
)

if(BUILD_BERTVITS2)
  include_directories(
  	${CMAKE_CURRENT_LIST_DIR}/include/bertvits2
  )
endif()

if(BUILD_PIPER)
  include_directories(
  	${CMAKE_CURRENT_LIST_DIR}/include/piper
    ${CMAKE_CURRENT_LIST_DIR}/third_party/piper/espeak-ng/src/include/espeak-ng/
  )
endif()

if(BUILD_SUPERTONIC)
  include_directories(
  	${CMAKE_CURRENT_LIST_DIR}/include/supertonic
  )
endif()

# 直接指定MNN库路径
set(MNN_LIB ${MNN_ROOT}/MNN)

# # 收集源文件
# file(GLOB_RECURSE SOURCES 
#     "src/*.cpp"
#     "src/*.cc"
# )

# # 排除示例目录
# list(FILTER SOURCES EXCLUDE REGEX "examples/")

if(BUILD_BERTVITS2)
    file(GLOB_RECURSE BERTVITS2_SOURCES 
        "src/bertvits2/*.cpp"
    )
    list(APPEND SOURCES ${BERTVITS2_SOURCES})
endif()

if(BUILD_PIPER)
    file(GLOB_RECURSE PIPER_SOURCES 
        "src/piper/*.cpp"
        "third_party/piper/espeak-ng/src/libespeak-ng/*.c"
    )
    list(APPEND SOURCES ${PIPER_SOURCES})
endif()

if(BUILD_SUPERTONIC)
    file(GLOB_RECURSE SUPERTONIC_SOURCES 
        "src/supertonic/*.cpp"
    )
    list(APPEND SOURCES ${SUPERTONIC_SOURCES})
endif()

# # 只有当需要构建时才包含相应的源文件
# if(NOT BUILD_BERTVITS2)
#     list(FILTER SOURCES EXCLUDE REGEX "src/bertvits2/")
# endif()

# if(NOT BUILD_PIPER)
#     list(FILTER SOURCES EXCLUDE REGEX "src/piper/")
# endif()

# if(NOT BUILD_SUPERTONIC)
#     list(FILTER SOURCES EXCLUDE REGEX "src/supertonic/")
# endif()

# # 创建共享库
# add_library(${PROJECT_NAME} SHARED ${SOURCES})

# # 链接MNN库
# target_link_libraries(${PROJECT_NAME} 
#     ${MNN_LIB}
# )

# 安装规则
# install(TARGETS ${PROJECT_NAME}
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
# )

# install(DIRECTORY include/ 
#     DESTINATION include
#     FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
# )

set(SHARED_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/mnn_tts_config.cpp
    ${CMAKE_SOURCE_DIR}/src/mnn_tts_logger.cpp
    ${CMAKE_SOURCE_DIR}/src/mnn_tts_sdk.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_SOURCE_DIR}/src/supertonic/mnn_supertonic_tts_impl.cpp
    ${CMAKE_SOURCE_DIR}/src/supertonic/mnn_supertonic_tts_impl.cpp
)
add_subdirectory(examples)