// This is slightly modified shader from slang playground

struct Vertex
{
    float3 position;
    float3 normal;
    float2 uv;
}

/*
struct Material
{
    int albedoTextureIndex;
    int normalTextureIndex;
}

struct NodeData {
    float4x4 transform;
}
*/
struct Scene
{
    //StructuredBuffer<Material> materials;
    //StructuredBuffer<NodeData> nodes;
    Texture2D textures[1];
    SamplerState samplers[1];
}


struct Camera
{
    float4x4 viewProjection;
}

ParameterBlock<Scene> scene;
[vk::push_constant] ConstantBuffer<Camera> camera;
//SamplerState samplerState;

struct VOut
{
    float4 position : SV_Position;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD;
}

[shader("vertex")]
VOut vertexMain(int instanceId: SV_InstanceID, Vertex input)
{
    VOut output;
    //output.position = mul(camera.viewProjection, mul(scene.nodes[instanceId].transform, float4(input.position, 1.0)));
    output.position = mul(camera.viewProjection, float4(input.position, 1.0));
    output.normal = input.normal;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VOut input, int instanceId: SV_InstanceID) : SV_Target
{
    let texture = scene.textures[0];//[scene.materials[instanceId].albedoTextureIndex];
    let color = texture.Sample(scene.samplers[0], input.uv);
    //let color = float4(input.normal * 0.5 + 0.5, 1.0);
    return color;
}
