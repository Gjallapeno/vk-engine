#include <engine/camera.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace engine {

glm::vec3 Camera::forward() const {
    return glm::normalize(glm::vec3(
        cosf(pitch) * cosf(yaw),
        sinf(pitch),
        cosf(pitch) * sinf(yaw)
    ));
}

glm::vec3 Camera::right() const {
    return glm::normalize(glm::cross(forward(), {0.0f, 1.0f, 0.0f}));
}

glm::mat4 Camera::view_projection(float aspect, float near, float far) const {
    glm::mat4 view = glm::lookAt(position, position + forward(), {0.0f, 1.0f, 0.0f});
    glm::mat4 proj = glm::perspective(glm::radians(90.0f), aspect, near, far);
    return proj * view;
}

} // namespace engine

