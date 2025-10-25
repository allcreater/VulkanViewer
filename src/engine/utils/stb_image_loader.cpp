module;
#include "stb_image.h"

export module utils.image_loader;
import std;
import utils.core;

export template <class ElementType>
struct shared_ownership_accessor {
    using offset_policy    = shared_ownership_accessor;
    using element_type     = ElementType;
    using reference        = ElementType&;
    using data_handle_type = std::shared_ptr<ElementType[]>;

    // from std::default_accessor
    static_assert(sizeof(element_type) > 0, "ElementType must be a complete type (N4950 [mdspan.accessor.default.overview]/2).");
    static_assert(!std::is_abstract_v<element_type>,"ElementType cannot be an abstract type (N4950 [mdspan.accessor.default.overview]/2).");
    static_assert(!std::is_array_v<element_type>, "ElementType cannot be an array type (N4950 [mdspan.accessor.default.overview]/2).");

    constexpr shared_ownership_accessor() noexcept = default;

    template <class OtherElementType>
        requires std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>
    constexpr shared_ownership_accessor(shared_ownership_accessor<OtherElementType>) noexcept {}

    // template <class OtherElementType>
    //     requires std::is_convertible_v<element_type (*)[], OtherElementType (*)[]>
    // constexpr operator std::default_accessor<OtherElementType>() {
    //     return std::default_accessor<OtherElementType>();
    // }

    [[nodiscard]] constexpr reference access(data_handle_type ptr, size_t i) const noexcept {
        return ptr[i];
    }

    [[nodiscard]] constexpr data_handle_type offset(data_handle_type ptr, size_t i) const noexcept {
        return data_handle_type{ptr, ptr.get() + i};
    }
};

// height, width, RGBA
using image2d_rgba_extents = std::extents<std::uint32_t, std::dynamic_extent, std::dynamic_extent, 4>;

export template <typename ElementType>
using image2d_rgba = std::mdspan<
    ElementType,
    image2d_rgba_extents,
    std::layout_right,
    shared_ownership_accessor<ElementType>
>;

export template <typename ElementType>
requires one_of_types<std::remove_cv_t<ElementType>, stbi_uc, stbi_us, float>
image2d_rgba<ElementType> loadImageAsRgba(std::span<const stbi_uc> imageRawData) {
    using MutableElementType = std::remove_cv_t<ElementType>;

    int width = 0;
    int height = 0;
    int channelsInFile = 0;

    ElementType* data = nullptr;
    if constexpr (sizeof (ElementType) == 1) {
        data = stbi_load_from_memory(imageRawData.data(), imageRawData.size(), &width, &height, &channelsInFile, STBI_rgb_alpha);
    } else if constexpr (sizeof (ElementType) == 2) {
        data = stbi_load_16_from_memory(imageRawData.data(), imageRawData.size(), &width, &height, &channelsInFile, STBI_rgb_alpha);
    } else if constexpr (sizeof (ElementType) == 4) {
        data = stbi_loadf_from_memory(imageRawData.data(), imageRawData.size(), &width, &height, &channelsInFile, STBI_rgb_alpha);
    }

    if (!data || width <= 0 || height <= 0)
        throw std::runtime_error(std::format("Failed to decode image: {}", stbi_failure_reason()));

    auto sharedData = std::shared_ptr<ElementType[]>(data, [](ElementType* ptr) {stbi_image_free(const_cast<MutableElementType*>(ptr));});
    return image2d_rgba<ElementType>(std::move(sharedData), image2d_rgba_extents{static_cast<std::size_t>(width), static_cast<std::size_t>(height)});
}

export template <typename ElementType>
requires one_of_types<std::remove_cv_t<ElementType>, stbi_uc, stbi_us, float>
image2d_rgba<ElementType> loadImageAsRgba(const std::filesystem::path& imagePath) {
    return loadImageAsRgba<ElementType>(readFile<stbi_uc>(imagePath));
}
