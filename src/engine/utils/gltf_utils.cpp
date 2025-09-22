module;
#include <tiny_gltf.h>
#include <cassert>

export module utils.gltf;

import std;

template <typename T, typename... Variants>
concept one_of_types = (std::is_same_v<T, Variants> || ...);

export {
    template <typename T>
    concept ElementaryAccessorType =
        one_of_types<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, float, double>;

    std::span<const std::byte> getAccessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor);

    // TODO: add support of any output ranges?
    template <ElementaryAccessorType T>
    void readAccessorDataTo(const tinygltf::Model& model, const tinygltf::Accessor& accessor, std::span<T> destination);

    template <ElementaryAccessorType T>
    std::vector<T> readAccessorDataAs(const tinygltf::Model& model, const tinygltf::Accessor& accessor);
}


// Implementation

template <typename Src, typename Dest, bool doNormalize>
void transform_fn(std::span<const std::byte> sourceData, std::span<std::byte> destinationData) {
    constexpr auto multiplier     = static_cast<Dest>(1.0) / std::numeric_limits<Dest>::max();
    const auto     elements_count = sourceData.size_bytes() / sizeof(Src);
    assert(destinationData.size_bytes() == elements_count * sizeof(Dest));

    if constexpr (std::is_same_v<std::decay_t<Src>, std::decay_t<Dest>>) {
        std::memcpy(destinationData.data(), sourceData.data(), sourceData.size_bytes());
        return;
    }

    const std::byte* pSourceValue      = sourceData.data();
    std::byte*       pDestinationValue = destinationData.data();
    for (size_t i = 0; i < elements_count; i++) {
        Src sourceValue;
        std::memcpy(&sourceValue, pSourceValue, sizeof(Src));

        Dest result = static_cast<Dest>(sourceValue);
        if constexpr (doNormalize) {
            result *= multiplier;
        }
        std::memcpy(pDestinationValue, &result, sizeof(Dest));

        pSourceValue += sizeof(Src);
        pDestinationValue += sizeof(Dest);
    }
};

template <ElementaryAccessorType T>
void readAccessorDataTo(const tinygltf::Model& model, const tinygltf::Accessor& accessor, std::span<T> destination) {
    assert(accessor.count > 0);
    assert((accessor.sparse.isSparse == false) && "sparse accessors are not implemented");
    assert(destination.size() == accessor.count * tinygltf::GetNumComponentsInType(accessor.type));

    const auto transform = [&accessor, data = getAccessorData(model, accessor), dst_bytes = std::as_writable_bytes(destination)]<bool doNormalize> {
        switch (accessor.componentType) {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
                return transform_fn<std::int8_t, T, doNormalize>(data, dst_bytes);
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                return transform_fn<std::uint8_t, T, doNormalize>(data, dst_bytes);
            case TINYGLTF_COMPONENT_TYPE_SHORT:
                return transform_fn<std::int16_t, T, doNormalize>(data, dst_bytes);
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                return transform_fn<std::uint16_t, T, doNormalize>(data, dst_bytes);
            case TINYGLTF_COMPONENT_TYPE_INT:
                return transform_fn<std::int32_t, T, doNormalize>(data, dst_bytes);
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                return transform_fn<std::uint32_t, T, doNormalize>(data, dst_bytes);
            case TINYGLTF_COMPONENT_TYPE_FLOAT:
                return transform_fn<float, T, doNormalize>(data, dst_bytes);
            case TINYGLTF_COMPONENT_TYPE_DOUBLE:
                return transform_fn<double, T, doNormalize>(data, dst_bytes);
        }
    };

    constexpr bool target_type_needs_normalization = std::is_floating_point_v<T>;
    if (accessor.normalized && target_type_needs_normalization) {
        assert((tinygltf::GetComponentSizeInBytes(accessor.normalized) <= 2) && "glTF 2.0: normalization is only supported for bytes and shorts");
        transform.operator()<true>();
    } else {
        transform.operator()<false>();
    }
}

template <ElementaryAccessorType T>
std::vector<T> readAccessorDataAs(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    std::vector<T> result(accessor.count * tinygltf::GetNumComponentsInType(accessor.type));
    readAccessorDataTo(model, accessor, std::span{result});
    return result;
}

std::span<const std::byte> getAccessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer     = model.buffers[bufferView.buffer];
    const auto* data       = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

    assert(accessor.ByteStride(bufferView) > 0);
    const size_t byteCount = accessor.count * accessor.ByteStride(bufferView);

    assert(byteCount <= bufferView.byteLength);
    assert((reinterpret_cast<uintptr_t>(data) % static_cast<uintptr_t>(tinygltf::GetComponentSizeInBytes(accessor.componentType))) == 0);
    return std::span{reinterpret_cast<const std::byte*>(data), byteCount};
}
