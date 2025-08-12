#include <engine/gfx/vulkan_pipeline.hpp>
#include <engine/vk_checks.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <vector>

namespace engine {

static std::vector<char> read_binary(const std::string& path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary);
  if (!f) { spdlog::error("[vk] Failed to open SPIR-V: {}", path); std::abort(); }
  size_t size = (size_t)f.tellg();
  std::vector<char> data(size);
  f.seekg(0);
  f.read(data.data(), size);
  return data;
}

VkShaderModule TrianglePipeline::load_module(const std::string& path) {
  auto bytes = read_binary(path);
  VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  ci.codeSize = bytes.size();
  ci.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
  VkShaderModule mod = VK_NULL_HANDLE;
  VK_CHECK(vkCreateShaderModule(dev_, &ci, nullptr, &mod));
  return mod;
}

TrianglePipeline::TrianglePipeline(const TrianglePipelineCreateInfo& ci)
  : dev_(ci.device)
{
  // Push constants: 1 float (aspect)
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pcr.offset = 0;
  pcr.size = sizeof(float);

  VkPipelineLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  lci.pushConstantRangeCount = 1;
  lci.pPushConstantRanges = &pcr;
  VK_CHECK(vkCreatePipelineLayout(dev_, &lci, nullptr, &layout_));

  VkShaderModule vs = load_module(ci.vs_spv);
  VkShaderModule fs = load_module(ci.fs_spv);

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs;
  stages[0].pName  = "main";
  stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName  = "main";

  VkPipelineVertexInputStateCreateInfo vin{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
  VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
  vp.viewportCount = 1;
  vp.scissorCount  = 1;

  VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
  rs.depthClampEnable = VK_FALSE;
  rs.rasterizerDiscardEnable = VK_FALSE;
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;                   // <- disable culling to avoid winding surprises
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;    // still set a sensible front face
  rs.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState cbAtt{};
  cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
  cb.attachmentCount = 1;
  cb.pAttachments = &cbAtt;

  VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
  VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
  dyn.dynamicStateCount = 2;
  dyn.pDynamicStates = dynStates;

  VkPipelineRenderingCreateInfo renderInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
  VkFormat colorFmt = ci.color_format;
  renderInfo.colorAttachmentCount = 1;
  renderInfo.pColorAttachmentFormats = &colorFmt;

  VkGraphicsPipelineCreateInfo gpc{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
  gpc.pNext = &renderInfo;
  gpc.stageCount = 2;
  gpc.pStages = stages;
  gpc.pVertexInputState   = &vin;
  gpc.pInputAssemblyState = &ia;
  gpc.pViewportState      = &vp;
  gpc.pRasterizationState = &rs;
  gpc.pMultisampleState   = &ms;
  gpc.pColorBlendState    = &cb;
  gpc.pDynamicState       = &dyn;
  gpc.layout = layout_;
  gpc.renderPass = VK_NULL_HANDLE;
  gpc.subpass = 0;

  VK_CHECK(vkCreateGraphicsPipelines(dev_, VK_NULL_HANDLE, 1, &gpc, nullptr, &pipeline_));

  vkDestroyShaderModule(dev_, fs, nullptr);
  vkDestroyShaderModule(dev_, vs, nullptr);
}

TrianglePipeline::~TrianglePipeline() {
  if (pipeline_) vkDestroyPipeline(dev_, pipeline_, nullptr);
  if (layout_)   vkDestroyPipelineLayout(dev_, layout_, nullptr);
}

} // namespace engine
