set(TARGET_NAME ${PROJECT_NAME}_app)

add_executable(${TARGET_NAME} main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE llama)