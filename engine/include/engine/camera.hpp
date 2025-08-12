#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace engine {

struct Camera {
    glm::vec3 position{0.0f, 0.0f, 2.0f};
    float yaw   = -glm::half_pi<float>();
    float pitch = 0.0f;

    glm::mat4 view_projection(float aspect, float near = 0.1f, float far = 100.0f) const;
    glm::vec3 forward() const;
    glm::vec3 right() const;
};

} // namespace engine

