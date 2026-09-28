struct FrameData {
    float4x4 viewProj;
};

struct PushData {
    float4x4 model;
};

[[vk::binding(0, 0)]]
ConstantBuffer<FrameData> frame;

[[vk::push_constant]]
ConstantBuffer<PushData> push;

struct VsIn {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float3 color : COLOR;
};

struct VsOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : NORMAL;
    [[vk::location(1)]] float3 color : COLOR;
};

struct FsIn {
    [[vk::location(0)]] float3 normal : NORMAL;
    [[vk::location(1)]] float3 color : COLOR;
};

[shader("vertex")]
VsOut vertexMain(VsIn input) {
    VsOut output;
    float4 world = mul(push.model, float4(input.position, 1.0));
    output.position = mul(frame.viewProj, world);
    output.normal = mul((float3x3)push.model, input.normal);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fragmentMain(FsIn input) : SV_Target {
    float3 normal = normalize(input.normal);
    float3 lightDir = normalize(float3(0.35, 0.85, 0.45));
    float diffuse = max(dot(normal, lightDir), 0.0);
    float3 lit = input.color * (0.35 + 0.65 * diffuse);
    return float4(lit, 1.0);
}
