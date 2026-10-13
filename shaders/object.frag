#version 450

layout(location = 0) in vec3 in_color;

layout(push_constant) uniform PushConstants {
	mat4 mvp;
	vec4 tint; // цвет, выбранный в интерфейсе
} pc;

layout(location = 0) out vec4 out_color;

void main() {
	// Процедурный цвет вершины умножается на выбранный в интерфейсе цвет.
	out_color = vec4(in_color * pc.tint.rgb, 1.0);
}
