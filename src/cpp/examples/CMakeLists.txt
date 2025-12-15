cmake_minimum_required(VERSION 3.13)
project(pixelai_tts_demo)

# 生成demo的配置文件，生成的可执行文件在build/examples/pixelai_tts_demo
# add_executable(pixelai_tts_demo demo.cpp)

# 添加Supertonic示例
add_executable(pixelai_tts_demo supertonic_demo.cpp)

target_sources(pixelai_tts_demo PRIVATE ${SHARED_SOURCE_FILES})

# # 包含目录
# target_include_directories(pixelai_tts_demo PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/third_party/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/third_party/include/nlohmann
# )

# target_include_directories(supertonic_demo PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/third_party/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/third_party/include/nlohmann
# )

if (WIN32)
	message(Windows)
elseif (APPLE)
	target_link_libraries(
		pixelai_tts_demo
		/Users/ws/ws/projects/sdk/mnn_sdks/MNN_All/MNN-MacOS-CPU-GPU/Dynamic/MNN.framework
	)
elseif (UNIX AND NOT APPLE)
	target_link_libraries(
		pixelai_tts_demo
		MNN
		MNN_Express
		dl
		pthread
	)
endif()