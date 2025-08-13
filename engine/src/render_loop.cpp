#include <engine/render_loop.hpp>
#include <engine/config.hpp>
#include <engine/gfx/present_pipeline.hpp>
#include <engine/gfx/ray_pipeline.hpp>
#include <engine/gfx/vulkan_instance.hpp>

namespace engine {

static constexpr float kColorCompute[4]  = {0.0f, 0.6f, 1.0f, 1.0f};
static constexpr float kColorGraphics[4] = {1.0f, 0.4f, 0.0f, 1.0f};
static constexpr float kColorTransfer[4] = {0.4f, 1.0f, 0.4f, 1.0f};

void record_present(VkCommandBuffer cmd, VkImage, VkImageView view,
                    VkFormat, VkExtent2D extent, void *user) {
  auto *ctx = static_cast<DrawCtx *>(user);
  auto begin_label = [&](const char *name, const float color[4]) {
    if (!cfg::kGpuMarkers || !vkCmdBeginDebugUtilsLabelEXT) return;
    VkDebugUtilsLabelEXT l{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    l.pLabelName = name;
    for (int i = 0; i < 4; ++i) l.color[i] = color[i];
    vkCmdBeginDebugUtilsLabelEXT(cmd, &l);
  };
  auto end_label = [&]() {
    if (cfg::kGpuMarkers && vkCmdEndDebugUtilsLabelEXT)
      vkCmdEndDebugUtilsLabelEXT(cmd);
  };

  begin_label("Voxel generation", kColorCompute);
  VkImageMemoryBarrier2 pre[3]{};
  for (int i = 0; i < 3; i++) {
    pre[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    pre[i].srcStageMask =
        ctx->first_frame ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    pre[i].srcAccessMask =
        ctx->first_frame ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    pre[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    pre[i].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    pre[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    pre[i].subresourceRange.levelCount = 1;
    pre[i].subresourceRange.layerCount = 1;
  }
  pre[0].image = ctx->occ_image;
  pre[0].oldLayout = ctx->occ_layout;
  pre[1].image = ctx->mat_image;
  pre[1].oldLayout = ctx->mat_layout;
  pre[2].image = ctx->occ_l1_image;
  pre[2].oldLayout = ctx->occ_l1_layout;

  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.imageMemoryBarrierCount = 3;
  dep.pImageMemoryBarriers = pre;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->occ_layout = VK_IMAGE_LAYOUT_GENERAL;
  ctx->mat_layout = VK_IMAGE_LAYOUT_GENERAL;
  ctx->occ_l1_layout = VK_IMAGE_LAYOUT_GENERAL;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_layout,
                          0, 1, &ctx->comp_set, 0, nullptr);
  uint32_t gx = (ctx->dispatch_dim.width + 7) / 8;
  uint32_t gy = (ctx->dispatch_dim.height + 7) / 8;
  uint32_t gz = (ctx->dispatch_dim.depth + 7) / 8;
  for (const auto &bp : ctx->brick_params) {
    vkCmdPushConstants(cmd, ctx->comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(BrickParams), &bp);
    vkCmdDispatch(cmd, gx, gy, gz);
  }
  end_label();

  begin_label("Build occupancy", kColorCompute);
  VkImageMemoryBarrier2 mid[2]{};
  for (int i = 0; i < 2; i++) {
    mid[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    mid[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mid[i].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mid[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mid[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    mid[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    mid[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    mid[i].subresourceRange.levelCount = 1;
    mid[i].subresourceRange.layerCount = 1;
  }
  mid[0].image = ctx->occ_image;
  mid[0].oldLayout = ctx->occ_layout;
  mid[1].image = ctx->mat_image;
  mid[1].oldLayout = ctx->mat_layout;
  dep.imageMemoryBarrierCount = 2;
  dep.pImageMemoryBarriers = mid;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->occ_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  ctx->mat_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkImage occ_images[3] = {ctx->occ_image, ctx->occ_l1_image, ctx->occ_l2_image};
  VkExtent3D occ_dims[3] = {ctx->occ_dim, ctx->occ_l1_dim, ctx->occ_l2_dim};
  VkImageLayout *occ_layouts[3] = {&ctx->occ_layout, &ctx->occ_l1_layout, &ctx->occ_l2_layout};

  for (uint32_t level = 0; level < ctx->occ_levels - 1; ++level) {
    VkImageMemoryBarrier2 b[2]{};
    for (int i = 0; i < 2; ++i) {
      b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
      b[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      b[i].subresourceRange.levelCount = 1;
      b[i].subresourceRange.layerCount = 1;
    }
    b[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b[0].oldLayout = *occ_layouts[level];
    b[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b[0].image = occ_images[level];

    b[1].srcStageMask = ctx->first_frame ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b[1].srcAccessMask = ctx->first_frame ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_WRITE_BIT;
    b[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b[1].dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b[1].oldLayout = *occ_layouts[level + 1];
    b[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[1].image = occ_images[level + 1];

    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers = b;
    vkCmdPipelineBarrier2(cmd, &dep);

    *occ_layouts[level] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    *occ_layouts[level + 1] = VK_IMAGE_LAYOUT_GENERAL;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_l1_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->comp_l1_layout, 0, 1,
                            &ctx->comp_l1_sets[level], 0, nullptr);

    BuildOccParams params{};
    params.srcDim = {static_cast<int>(occ_dims[level].width),
                     static_cast<int>(occ_dims[level].height),
                     static_cast<int>(occ_dims[level].depth)};
    params.dstDim = {static_cast<int>(occ_dims[level + 1].width),
                     static_cast<int>(occ_dims[level + 1].height),
                     static_cast<int>(occ_dims[level + 1].depth)};
    params.blockSize = static_cast<int>(occ_dims[level].width /
                                        occ_dims[level + 1].width);
    vkCmdPushConstants(cmd, ctx->comp_l1_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(params), &params);

    gx = (occ_dims[level + 1].width + 7) / 8;
    gy = (occ_dims[level + 1].height + 7) / 8;
    gz = (occ_dims[level + 1].depth + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, gz);
  }
  end_label();

  begin_label("Raycast", kColorGraphics);
  VkRenderingAttachmentInfo gatt[3]{};
  VkImageView gviews[3] = {ctx->g_albedo_view, ctx->g_normal_view, ctx->g_depth_view};
  VkImageLayout glayouts[3] = {ctx->g_albedo_layout, ctx->g_normal_layout, ctx->g_depth_layout};
  for (int i = 0; i < 3; ++i) {
    gatt[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    gatt[i].imageView = gviews[i];
    gatt[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    gatt[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    gatt[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkClearValue c;
    c.color = {{0, 0, 0, 0}};
    gatt[i].clearValue = c;
    glayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
  }

  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea.offset = {0, 0};
  ri.renderArea.extent = ctx->ray_extent;
  ri.layerCount = 1;
  ri.colorAttachmentCount = 3;
  ri.pColorAttachments = gatt;

  VkViewport vp{};
  vp.x = 0.0f;
  vp.y = static_cast<float>(ctx->ray_extent.height);
  vp.width = static_cast<float>(ctx->ray_extent.width);
  vp.height = -static_cast<float>(ctx->ray_extent.height);
  vp.minDepth = 0.0f;
  vp.maxDepth = 1.0f;

  VkRect2D sc{{0, 0}, ctx->ray_extent};

  vkCmdBeginRendering(cmd, &ri);
  vkCmdSetViewport(cmd, 0, 1, &vp);
  vkCmdSetScissor(cmd, 0, 1, &sc);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    ctx->ray_pipe->pipeline());
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          ctx->ray_pipe->layout(), 0, 1, &ctx->ray_dset, 0,
                          nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);
  end_label();

  begin_label("Readback", kColorTransfer);
  VkImageMemoryBarrier2 gpre[3]{};
  VkImageLayout *g_layouts[3] = {&ctx->g_albedo_layout, &ctx->g_normal_layout,
                                 &ctx->g_depth_layout};
  VkImage g_images[3] = {ctx->g_albedo, ctx->g_normal, ctx->g_depth};
  for (int i = 0; i < 3; i++) {
    gpre[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    gpre[i].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    gpre[i].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    gpre[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    gpre[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    gpre[i].oldLayout = *g_layouts[i];
    gpre[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    gpre[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    gpre[i].subresourceRange.levelCount = 1;
    gpre[i].subresourceRange.layerCount = 1;
    gpre[i].image = g_images[i];
  }
  dep.imageMemoryBarrierCount = 3;
  dep.pImageMemoryBarriers = gpre;
  vkCmdPipelineBarrier2(cmd, &dep);

  for (int i = 0; i < 3; ++i) *g_layouts[i] = VK_IMAGE_LAYOUT_GENERAL;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->comp_layout,
                          0, 1, &ctx->comp_set, 0, nullptr);
  gx = (ctx->dispatch_dim.width + 7) / 8;
  gy = (ctx->dispatch_dim.height + 7) / 8;
  gz = (ctx->dispatch_dim.depth + 7) / 8;
  for (const auto &bp : ctx->brick_params) {
    vkCmdPushConstants(cmd, ctx->comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(BrickParams), &bp);
    vkCmdDispatch(cmd, gx, gy, gz);
  }
  end_label();

  begin_label("Prepare readback", kColorTransfer);
  VkImageSubresourceRange stepsRange{};
  stepsRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  stepsRange.levelCount = 1;
  stepsRange.layerCount = 1;

  VkImageMemoryBarrier2 steps_to_copy{};
  steps_to_copy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  steps_to_copy.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  steps_to_copy.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  steps_to_copy.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  steps_to_copy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  steps_to_copy.oldLayout = ctx->steps_layout;
  steps_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  steps_to_copy.subresourceRange = stepsRange;
  steps_to_copy.image = ctx->steps_image;

  VkImageMemoryBarrier2 gpost[3]{};
  VkImageLayout *g_layouts2[3] = {&ctx->g_albedo_layout, &ctx->g_normal_layout,
                                  &ctx->g_depth_layout};
  VkImage g_images2[3] = {ctx->g_albedo, ctx->g_normal, ctx->g_depth};
  for (int i = 0; i < 3; i++) {
    gpost[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    gpost[i].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    gpost[i].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    gpost[i].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    gpost[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    gpost[i].oldLayout = *g_layouts2[i];
    gpost[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    gpost[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    gpost[i].subresourceRange.levelCount = 1;
    gpost[i].subresourceRange.layerCount = 1;
    gpost[i].image = g_images2[i];
  }

  VkImageMemoryBarrier2 readback_barriers[4] = {steps_to_copy, gpost[0], gpost[1], gpost[2]};
  dep.imageMemoryBarrierCount = 4;
  dep.pImageMemoryBarriers = readback_barriers;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->steps_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  for (int i = 0; i < 3; ++i)
    *g_layouts2[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkBufferImageCopy bic{};
  bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  bic.imageSubresource.layerCount = 1;
  bic.imageExtent = {ctx->steps_dim.width, ctx->steps_dim.height, 1};
  vkCmdCopyImageToBuffer(cmd, ctx->steps_image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         ctx->steps_buffer, 1, &bic);

  VkImageMemoryBarrier2 steps_to_clear{};
  steps_to_clear.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  steps_to_clear.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  steps_to_clear.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  steps_to_clear.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
  steps_to_clear.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  steps_to_clear.oldLayout = ctx->steps_layout;
  steps_to_clear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  steps_to_clear.subresourceRange = stepsRange;
  steps_to_clear.image = ctx->steps_image;
  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &steps_to_clear;
  vkCmdPipelineBarrier2(cmd, &dep);

  ctx->steps_layout = VK_IMAGE_LAYOUT_GENERAL;

  VkClearColorValue zero{{0, 0, 0, 0}};
  vkCmdClearColorImage(cmd, ctx->steps_image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1,
                       &stepsRange);
  end_label();

  begin_label("Postprocess", kColorGraphics);
  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkClearValue clear;
  clear.color = {{0.06f, 0.07f, 0.10f, 1.0f}};
  color.clearValue = clear;

  VkRenderingInfo ri2{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri2.renderArea.offset = {0, 0};
  ri2.renderArea.extent = extent;
  ri2.layerCount = 1;
  ri2.colorAttachmentCount = 1;
  ri2.pColorAttachments = &color;

  vkCmdBeginRendering(cmd, &ri2);

  VkViewport vp_hi{};
  vp_hi.x = 0.0f;
  vp_hi.y = static_cast<float>(extent.height);
  vp_hi.width = static_cast<float>(extent.width);
  vp_hi.height = -static_cast<float>(extent.height);
  vp_hi.minDepth = 0.0f;
  vp_hi.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp_hi);

  VkRect2D sc_hi{{0, 0}, extent};
  vkCmdSetScissor(cmd, 0, 1, &sc_hi);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    ctx->light_pipe->pipeline());
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          ctx->light_pipe->layout(), 0, 1, &ctx->light_dset, 0,
                          nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);
  end_label();

  ctx->first_frame = false;
}

} // namespace engine

