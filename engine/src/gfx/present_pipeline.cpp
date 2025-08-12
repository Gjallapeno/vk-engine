#include <engine/gfx/present_pipeline.hpp>
#include <engine/vk_checks.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <vector>

namespace engine {

static std::vector<char> read_binary(const std::string& path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary);
  if (!f) { spdlog::error("[vk] Failed to open SPIR-V: {}", path); std::abort(); }
  size_t size = static_cast<size_t>(f.tellg());
  std::vector<char> data(size);
  f.seekg(0);
  f.read(data.data(), size);
  return data;
}

VkShaderModule PresentPipeline::load_module(const std::string& path) {
  auto bytes = read_binary(path);
  VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  ci.codeSize = bytes.size();
  ci.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
  VkShaderModule mod = VK_NULL_HANDLE;
  VK_CHECK(vkCreateShaderModule(dev_, &ci, nullptr, &mod));
  return mod;
}

PresentPipeline::PresentPipeline(const PresentPipelineCreateInfo& ci)
  : dev_(ci.device), color_format_(ci.color_format) {
  // Descriptor set layout for camera UBO, voxel AABB UBO and voxel textures
  VkDescriptorSetLayoutBinding binds[4]{};
  binds[0].binding = 0;
  binds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  binds[0].descriptorCount = 1;
  binds[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  binds[1].binding = 1;
  binds[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  binds[1].descriptorCount = 1;
  binds[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  binds[2].binding = 2;
  binds[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binds[2].descriptorCount = 1;
  binds[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  binds[3].binding = 3;
  binds[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binds[3].descriptorCount = 1;
  binds[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo dlci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
  dlci.bindingCount = 4; dlci.pBindings = binds;
  VK_CHECK(vkCreateDescriptorSetLayout(dev_, &dlci, nullptr, &dset_layout_));

  // Pipeline layout: only descriptor set layout (no push constants)
  VkPipelineLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  lci.setLayoutCount = 1; lci.pSetLayouts = &dset_layout_;
  VK_CHECK(vkCreatePipelineLayout(dev_, &lci, nullptr, &layout_));

  // Shaders
  VkShaderModule vs = load_module(ci.vs_spv);
  VkShaderModule fs = load_module(ci.fs_spv);

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName = "main";

  // No vertex inputs
  VkPipelineVertexInputStateCreateInfo vin{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

  VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
  vp.viewportCount = 1; vp.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
  ds.depthTestEnable = VK_FALSE;
  ds.depthWriteEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState cbAtt{};
  cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
  cb.attachmentCount = 1; cb.pAttachments = &cbAtt;

  VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
  VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
  dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynStates;

  // Dynamic rendering format
  VkPipelineRenderingCreateInfo renderInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
  VkFormat colorFmt = color_format_;
  renderInfo.colorAttachmentCount = 1;
  renderInfo.pColorAttachmentFormats = &colorFmt;
  renderInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

  VkGraphicsPipelineCreateInfo gpc{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
  gpc.pNext = &renderInfo;
  gpc.stageCount = 2; gpc.pStages = stages;
  gpc.pVertexInputState = &vin;
  gpc.pInputAssemblyState = &ia;
  gpc.pViewportState = &vp;
  gpc.pRasterizationState = &rs;
  gpc.pMultisampleState = &ms;
  gpc.pColorBlendState = &cb;
  gpc.pDepthStencilState = &ds;
  gpc.pDynamicState = &dyn;
  gpc.layout = layout_;
  gpc.renderPass = VK_NULL_HANDLE; gpc.subpass = 0;

  VK_CHECK(vkCreateGraphicsPipelines(dev_, ci.pipeline_cache, 1, &gpc, nullptr, &pipeline_));

  vkDestroyShaderModule(dev_, fs, nullptr);
  vkDestroyShaderModule(dev_, vs, nullptr);
}

PresentPipeline::~PresentPipeline() {
  if (pipeline_)    vkDestroyPipeline(dev_, pipeline_, nullptr);
  if (layout_)      vkDestroyPipelineLayout(dev_, layout_, nullptr);
  if (dset_layout_) vkDestroyDescriptorSetLayout(dev_, dset_layout_, nullptr);
}

} // namespace engine
