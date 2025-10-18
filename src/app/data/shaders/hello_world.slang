struct Vertex {
    float2 position;
    float4 color;
};

struct VertexStageOutput {
    float4 position     : SV_Position;
    float4 color;
};


[shader("vertex")]
VertexStageOutput vs_main(/*uint instanceId: SV_InstanceID, */Vertex input)
{
    VertexStageOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;

    return output;
}

[shader("fragment")]
float4 fs_main(VertexStageOutput input) : SV_Target
{
    return input.color;
}