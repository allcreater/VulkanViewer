module;
#include <glm/glm.hpp>

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
// #include <fastgltf/glm_element_traits.hpp>

export module engine:vulkan.renderer;
import vulkan_hpp;
import std;
import fastgltf;

import utils.core;
import :vulkan.context;

struct Synchronization {
    explicit Synchronization(const vk::raii::Device& device)
        : imageAvailable{device, vk::SemaphoreCreateInfo{}}
        , rederFinished{device, vk::SemaphoreCreateInfo{}}
        , inFlightFence{device, {.flags = vk::FenceCreateFlagBits::eSignaled}} {}

    vk::raii::Semaphore imageAvailable;
    vk::raii::Semaphore rederFinished;
    vk::raii::Fence     inFlightFence;
};

export class VulkanRenderer final {
public:
    VulkanRenderer(VulkanGraphicsContext&& graphicsContext);
    ~VulkanRenderer();

    void Render();
    void LoadModel(const std::filesystem::path& path);
private:
    void CreatePipeline();

private:
    constexpr static int num_inflight_frames = 2;

    VulkanGraphicsContext               graphicsContext;
    vk::raii::ShaderModule              shaderModule{nullptr};
    vk::raii::PipelineLayout            pipelineLayout{nullptr};
    vk::raii::Pipeline                  trianglePipeline{nullptr};
    ResourceFactory::Handle<vk::Buffer> vertexBuffer, indexBuffer;
    // vk::raii::CommandPool commandPool;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<Synchronization>         sync;
    uint32_t                             currentFrame = 0;
};

namespace
{

vk::raii::ShaderModule loadShaderModule(const vk::raii::Device& device, const std::filesystem::path& path) {
    const auto  data = readFile<std::uint32_t>(path);

    vk::ShaderModuleCreateInfo create_info{
        .flags    = {},
        .codeSize = std::span{data}.size_bytes(),
        .pCode    = data.data(),
    };

    return device.createShaderModule(create_info);
}

struct Vertex {
    fastgltf::math::fvec3 position;
    fastgltf::math::fvec3 normal;
    fastgltf::math::fvec2 uv;
};

struct Camera{
    glm::fmat4x4 viewProjection;
};

} // namespace


void VulkanRenderer::CreatePipeline() {
    const auto& device = graphicsContext.getDevice();
    shaderModule = loadShaderModule(device, "data/shaders/simple_model.spv");

    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .flags               = {},
            .stage               = vk::ShaderStageFlagBits::eVertex,
            .module              = shaderModule,
            .pName               = "vertexMain",
            .pSpecializationInfo = nullptr,
        },
        vk::PipelineShaderStageCreateInfo{
            .flags               = {},
            .stage               = vk::ShaderStageFlagBits::eFragment,
            .module              = shaderModule,
            .pName               = "fragmentMain",
            .pSpecializationInfo = nullptr,
        },
    };

    const std::array vertexBindingDescriptions{
        vk::VertexInputBindingDescription{
            .binding   = 0,
            .stride    = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex,
        },
    };
    constexpr std::array vertexAttributeDescriptions{
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding  = 0,
            .format   = vk::Format::eR32G32B32Sfloat,
            .offset   = offsetof(Vertex, position),
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding  = 0,
            .format   = vk::Format::eR32G32B32Sfloat,
            .offset   = offsetof(Vertex, normal),
        },
        vk::VertexInputAttributeDescription{
            .location = 2,
            .binding  = 0,
            .format   = vk::Format::eR32G32Sfloat,
            .offset   = offsetof(Vertex, uv),
        },
    };

    const vk::PipelineVertexInputStateCreateInfo vertexInputState{
        .flags                           = {},
        .vertexBindingDescriptionCount   = vertexBindingDescriptions.size(),
        .pVertexBindingDescriptions      = vertexBindingDescriptions.data(),
        .vertexAttributeDescriptionCount = vertexAttributeDescriptions.size(),
        .pVertexAttributeDescriptions    = vertexAttributeDescriptions.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{
        .flags                  = {},
        .topology               = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False,
    };

    const vk::PipelineTessellationStateCreateInfo tesselationState{
        .flags              = {},
        .patchControlPoints = 0,
    };

    const auto         swapChainExtent = graphicsContext.getExtent();
    const vk::Viewport viewport{
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = (float)swapChainExtent.width,
        .height   = (float)swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapChainExtent,
    };

    const vk::PipelineViewportStateCreateInfo viewportState{
        .flags         = {},
        .viewportCount = 1,
        .pViewports    = &viewport,
        .scissorCount  = 1,
        .pScissors     = &scissor,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizationState{
        .flags                   = {},
        .depthClampEnable        = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode             = vk::PolygonMode::eFill,
        .cullMode                = vk::CullModeFlagBits::eBack,
        .frontFace               = vk::FrontFace::eClockwise,
        .depthBiasEnable         = vk::False,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp          = 0.0f,
        .depthBiasSlopeFactor    = 0.0f,
        .lineWidth               = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        .flags                 = {},
        .rasterizationSamples  = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable   = vk::False,
        .minSampleShading      = 1.0f,
        .pSampleMask           = nullptr,
        .alphaToCoverageEnable = vk::False,
        .alphaToOneEnable      = vk::False,
    };

    // const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
    //	.flags                 = {},
    //	.depthTestEnable       = vk::False,
    //	.depthWriteEnable      = vk::False,
    //	.depthCompareOp        = vk::CompareOp::eAlways,
    //	.depthBoundsTestEnable = {},
    //	.stencilTestEnable     = {},
    //	.front                 = {},
    //	.back                  = {},
    //	.minDepthBounds        = {},
    //	.maxDepthBounds        = {},
    // };

    constexpr std::array<vk::DynamicState, 2> dynamicStates{
        vk::DynamicState::eViewport, vk::DynamicState::eScissor,
        // vk::DynamicState::eViewportWithCount,
        // vk::DynamicState::eScissorWithCount,
    };
    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .flags             = {},
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data(),
    };

    const std::array<vk::PipelineColorBlendAttachmentState, 1> colorBlendAttachments{
        vk::PipelineColorBlendAttachmentState{
            .blendEnable         = vk::False,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eZero,
            .colorBlendOp        = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp        = vk::BlendOp::eAdd,
            .colorWriteMask =
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        },
    };

    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .flags           = {},
        .logicOpEnable   = {},
        .logicOp         = vk::LogicOp::eClear,
        .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
        .pAttachments    = colorBlendAttachments.data(),
        .blendConstants  = std::array{0.0f, 0.0f, 0.0f, 0.0f},
    };

    // std::array bindings{
    //     vk::DescriptorSetLayoutBinding{
    //         .binding            = 0,
    //         .descriptorType     = vk::DescriptorType::eCombinedImageSampler,
    //         .descriptorCount    = 1,
    //         .stageFlags         = vk::ShaderStageFlagBits::eFragment,
    //         .pImmutableSamplers = nullptr,
    //     },
    // };
    //
    // vk::DescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo{
    //     .flags        = {},
    //     .bindingCount = bindings.size(),
    //     .pBindings    = bindings.data(),
    // };
    //
    // auto setLayout = device.createDescriptorSetLayout(descriptorLayoutCreateInfo);
    // const std::array setLayouts{
    //     *setLayout,
    // };

    constexpr std::array pushConstantsRanges{
        vk::PushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .offset     = 0,
            .size       = sizeof(glm::mat4),
        },
    };

    const vk::PipelineLayoutCreateInfo layoutCreateInfo{
        .flags                  = {},
        // .setLayoutCount         = setLayouts.size(),
        // .pSetLayouts            = setLayouts.data(),
        .pushConstantRangeCount = pushConstantsRanges.size(),
        .pPushConstantRanges    = pushConstantsRanges.data(),
    };
    pipelineLayout = device.createPipelineLayout(layoutCreateInfo);


    const std::array colorAttachmentFormats{graphicsContext.getSurfaceFormat().format};

    vk::StructureChain createInfo{vk::GraphicsPipelineCreateInfo{
                                      .flags               = {},
                                      .stageCount          = static_cast<uint32_t>(stages.size()),
                                      .pStages             = stages.data(),
                                      .pVertexInputState   = &vertexInputState,
                                      .pInputAssemblyState = &inputAssemblyState,
                                      .pTessellationState  = &tesselationState,
                                      .pViewportState      = &viewportState,
                                      .pRasterizationState = &rasterizationState,
                                      .pMultisampleState   = &multisampleState,
                                      .pDepthStencilState  = nullptr,
                                      .pColorBlendState    = &colorBlendState,
                                      .pDynamicState       = &dynamicState,
                                      .layout              = pipelineLayout,
                                      .subpass             = 0,
                                      .basePipelineHandle  = {nullptr},
                                      .basePipelineIndex   = -1,
                                  },
                                  vk::PipelineRenderingCreateInfo{
                                      .colorAttachmentCount    = static_cast<uint32_t>(colorAttachmentFormats.size()),
                                      .pColorAttachmentFormats = colorAttachmentFormats.data(),
                                  }};

    trianglePipeline = device.createGraphicsPipeline({nullptr}, createInfo.get());
}

VulkanRenderer::VulkanRenderer(VulkanGraphicsContext&& _graphicsContext) : graphicsContext{std::move(_graphicsContext)}, vertexBuffer{nullptr} {
    sync = std::views::iota(0, num_inflight_frames) | std::views::transform([&](auto _) { return Synchronization{graphicsContext.getDevice()}; }) |
        std::ranges::to<std::vector>();

    CreatePipeline();

    commandBuffers = graphicsContext.createCommandBuffers(num_inflight_frames);
}

VulkanRenderer::~VulkanRenderer() {
    const vk::raii::Device& device = graphicsContext.getDevice();
    device.waitIdle();
}

void VulkanRenderer::Render() {
    const vk::raii::Device&        device          = graphicsContext.getDevice();
    const vk::raii::CommandBuffer& commandBuffer   = commandBuffers[currentFrame];
    const auto                     swapChainExtent = graphicsContext.getExtent();
    const auto&                    frameSync       = sync[currentFrame];

    device.waitForFences(std::array{*frameSync.inFlightFence}, vk::True, std::numeric_limits<uint64_t>::max());
    device.resetFences(std::array{*frameSync.inFlightFence});

    const vk::raii::SwapchainKHR& swapchain = graphicsContext.getSwapchainData().swapchain;
    const auto [success, frameImageIndex]   = swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), frameSync.imageAvailable, nullptr);


    commandBuffer.reset();
    {
        commandBuffer.begin({});

        {
            const vk::ImageMemoryBarrier2 barrier{
                .srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
                .dstStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout     = vk::ImageLayout::eUndefined,
                .newLayout     = vk::ImageLayout::eColorAttachmentOptimal,
                .image         = graphicsContext.getSwapchainData().images[frameImageIndex],
                .subresourceRange =
                    {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
            };
            commandBuffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
        }

        const std::array colorAttachments{vk::RenderingAttachmentInfo{.imageView   = graphicsContext.getSwapchainData().imageViews[frameImageIndex],
                                                                      .imageLayout = vk::ImageLayout::eAttachmentOptimal,
                                                                      .loadOp      = vk::AttachmentLoadOp::eClear,
                                                                      .storeOp     = vk::AttachmentStoreOp::eStore,
                                                                      .clearValue  = {vk::ClearColorValue{1.0f, 1.0f, 0.0f, 1.0f}}}};
        commandBuffer.beginRendering({
            .renderArea           = {.offset = {}, .extent = swapChainExtent},
            .layerCount           = 1,
            .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
            .pColorAttachments    = colorAttachments.data(),
        });

        const std::array viewports{
            vk::Viewport{
                .x        = 0.0f,
                .y        = 0.0f,
                .width    = (float)swapChainExtent.width,
                .height   = (float)swapChainExtent.height,
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            },
        };
        commandBuffer.setViewport(0, viewports);


        const std::array<vk::Rect2D, 1> scissorRects{
            vk::Rect2D{
                .offset = {0, 0},
                .extent = swapChainExtent,
            },
        };
        commandBuffer.setScissor(0, scissorRects);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, trianglePipeline);

        if (vertexBuffer && indexBuffer) {
            commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
            commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);


            auto proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 100.0f);
            auto view = glm::lookAt(glm::vec3(20.0f, 20.0f, 20.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            Camera camera{
                .viewProjection = proj * view,
            };
            commandBuffer.pushConstants<Camera>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, vk::ArrayProxy{camera});

            int count = indexBuffer.mappedMemory().size() / sizeof(uint32_t);
            commandBuffer.drawIndexed(count, 1, 0, 0, 0);
            //commandBuffer.drawIndirect()
        }

        commandBuffer.endRendering();

        {
            const vk::ImageMemoryBarrier2 barrier{
                .srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eNone,
                .dstAccessMask = vk::AccessFlagBits2::eNone,
                .oldLayout     = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout     = vk::ImageLayout::ePresentSrcKHR,
                .image         = graphicsContext.getSwapchainData().images[frameImageIndex],
                .subresourceRange =
                    {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
            };
            commandBuffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
        }

        commandBuffer.end();
    }

    const std::array<vk::Semaphore, 1>          semaphoresForWait   = {frameSync.imageAvailable};
    const std::array<vk::Semaphore, 1>          semaphoresForSignal = {frameSync.rederFinished};
    const std::array<vk::PipelineStageFlags, 1> waitStages          = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
    const vk::SubmitInfo                        submitInfo{
                               .waitSemaphoreCount   = static_cast<uint32_t>(semaphoresForWait.size()),
                               .pWaitSemaphores      = semaphoresForWait.data(),
                               .pWaitDstStageMask    = waitStages.data(),
                               .commandBufferCount   = 1,
                               .pCommandBuffers      = &*commandBuffer,
                               .signalSemaphoreCount = static_cast<uint32_t>(semaphoresForSignal.size()),
                               .pSignalSemaphores    = semaphoresForSignal.data(),
    };

    graphicsContext.getGraphicsQueue().submit(submitInfo, frameSync.inFlightFence);

    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &*frameSync.rederFinished,
        .swapchainCount     = 1,
        .pSwapchains        = &*swapchain,
        .pImageIndices      = &frameImageIndex,
        .pResults           = nullptr,
    };

    graphicsContext.getPresentQueue().presentKHR(presentInfo);

    currentFrame = (currentFrame + 1) % num_inflight_frames;
}

template <typename T>
auto iterateAccessor (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive, std::string_view name) {
    auto it = primitive.findAttribute(name);
    if (it == primitive.attributes.end())
        throw std::runtime_error{"Model does not contain attribute"};

    return fastgltf::iterateAccessor<T>(asset, asset.accessors[it->accessorIndex]);
};

template <typename T, std::ranges::range R>
requires std::is_standard_layout_v<T>
auto writeRangeToMemory(R&& dataRange, std::span<std::byte> destMemory) {
    std::ranges::for_each(dataRange, [&destMemory](const T& vertex) {
        std::memcpy(destMemory.data(), &vertex, sizeof(T));
        destMemory = destMemory.subspan(sizeof(T));
    });
}

// template <typename... Args>
// auto iterateAccessors (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive, std::array<std::string_view, sizeof...(Args)> name) {
//
// }

void VulkanRenderer::LoadModel(const std::filesystem::path& path) {
    fastgltf::GltfFileStream stream{path};

    fastgltf::Parser parser{{}};
    auto result = parser.loadGltf(stream, path.parent_path(), fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages);
    if (!result) {
        throw std::runtime_error{std::format("Failed to load model: {}", fastgltf::getErrorName(result.error()))};
    }

    // make vertex buffer

    //result->meshes[0].primitives[0]

    const auto& primitive = result->meshes[0].primitives[0];

    // vertex
    {
        auto positions = iterateAccessor<fastgltf::math::fvec3>(result.get(), primitive, "POSITION");
        auto normals = iterateAccessor<fastgltf::math::fvec3>(result.get(), primitive, "NORMAL");
        auto uvs = iterateAccessor<fastgltf::math::fvec2>(result.get(), primitive, "TEXCOORD_0");
        auto interleavedVertexAttribs = std::views::zip(positions, normals, uvs) | std::views::transform([](auto&& tuple) {
            return std::make_from_tuple<Vertex>(tuple);
        });


        auto buffer = graphicsContext.getResourceFactory().CreateBuffer(
            {
                .flags       = {},
                .size        = std::ranges::distance(interleavedVertexAttribs) * sizeof(Vertex),
                .usage       = vk::BufferUsageFlagBits::eVertexBuffer,
                .sharingMode = vk::SharingMode::eExclusive,
            },
            CasualUsage::AutoMapped);

        writeRangeToMemory<Vertex>(interleavedVertexAttribs, buffer.mappedMemory());
        std::span target {reinterpret_cast<Vertex*>(buffer.mappedMemory().data()), buffer.mappedMemory().size_bytes() / sizeof(Vertex)};

        vertexBuffer = std::move(buffer);
    }

    // indices
    {
        const auto& accessor = result->accessors[primitive.indicesAccessor.value()];
        auto buffer = graphicsContext.getResourceFactory().CreateBuffer(
           {
               .flags       = {},
               .size        = accessor.count * sizeof(std::uint32_t),
               .usage       = vk::BufferUsageFlagBits::eIndexBuffer,
               .sharingMode = vk::SharingMode::eExclusive,
           },
           CasualUsage::AutoMapped);

        fastgltf::copyFromAccessor<uint32_t>(result.get(), accessor, buffer.mappedMemory().data());

        std::span target {reinterpret_cast<uint32_t*>(buffer.mappedMemory().data()), buffer.mappedMemory().size_bytes() / sizeof(uint32_t)};

        indexBuffer = std::move(buffer);
    }
}
