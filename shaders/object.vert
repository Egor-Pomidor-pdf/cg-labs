#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(push_constant) uniform PushConstants {
	mat4 mvp; // projection * view * model
} pc;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = pc.mvp * vec4(in_position, 1.0);
	out_color = in_color;
}
