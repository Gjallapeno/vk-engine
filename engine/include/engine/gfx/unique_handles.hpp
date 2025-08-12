#pragma once
#include <vulkan/vulkan.h>

namespace engine {

// Generic RAII wrapper for Vulkan objects tied to a VkDevice.
// DestroyFn must be a function pointer like vkDestroyPipeline.

template <typename T, void (*DestroyFn)(VkDevice, T, const VkAllocationCallbacks*)>
class UniqueHandle {
public:
  UniqueHandle() = default;
  explicit UniqueHandle(VkDevice device, T handle = VK_NULL_HANDLE)
      : dev_(device), handle_(handle) {}
  ~UniqueHandle() { reset(); }

  UniqueHandle(const UniqueHandle&) = delete;
  UniqueHandle& operator=(const UniqueHandle&) = delete;

  UniqueHandle(UniqueHandle&& other) noexcept { move_from(other); }
  UniqueHandle& operator=(UniqueHandle&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }

  T get() const { return handle_; }
  VkDevice device() const { return dev_; }
  explicit operator bool() const { return handle_ != VK_NULL_HANDLE; }

  // Prepare the wrapper to receive a newly created handle.
  T* init(VkDevice device) {
    reset();
    dev_ = device;
    return &handle_;
  }

  // Release ownership of the handle without destroying it.
  T release() {
    T tmp = handle_;
    handle_ = VK_NULL_HANDLE;
    dev_ = VK_NULL_HANDLE;
    return tmp;
  }

  // Destroy the handle if valid.
  void reset() {
    if (handle_ != VK_NULL_HANDLE) {
      DestroyFn(dev_, handle_, nullptr);
      handle_ = VK_NULL_HANDLE;
    }
    dev_ = VK_NULL_HANDLE;
  }

private:
  void move_from(UniqueHandle& other) {
    dev_ = other.dev_;
    handle_ = other.handle_;
    other.dev_ = VK_NULL_HANDLE;
    other.handle_ = VK_NULL_HANDLE;
  }

  VkDevice dev_ = VK_NULL_HANDLE;
  T handle_ = VK_NULL_HANDLE;
};

using UniquePipeline = UniqueHandle<VkPipeline, vkDestroyPipeline>;
using UniquePipelineLayout = UniqueHandle<VkPipelineLayout, vkDestroyPipelineLayout>;
using UniqueDescriptorPool = UniqueHandle<VkDescriptorPool, vkDestroyDescriptorPool>;
using UniqueDescriptorSetLayout =
    UniqueHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>;
using UniqueImageView = UniqueHandle<VkImageView, vkDestroyImageView>;
using UniqueSampler = UniqueHandle<VkSampler, vkDestroySampler>;

} // namespace engine

