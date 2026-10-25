#pragma once

#include <cmath>

// Minimal linear algebra for the labs.
//
// Mat4 stores its elements in COLUMN-MAJOR order, exactly like GLSL's mat4,
// so a Mat4 can be copied into a uniform/push constant block as raw bytes.
// c[j][i] is the element in row i, column j.

namespace linalg {

constexpr float pi = 3.14159265358979323846f;

inline float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

inline float degrees(float radians) {
	return radians * (180.0f / pi);
}

struct Vec3 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

inline float dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(Vec3 a, Vec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline float length(Vec3 v) {
	return std::sqrt(dot(v, v));
}

inline Vec3 normalize(Vec3 v) {
	const float len = length(v);
	return (len > 0.0f) ? v * (1.0f / len) : v;
}

struct Mat4 {
	float c[4][4] = {};
};

inline Mat4 identity() {
	Mat4 m;
	m.c[0][0] = m.c[1][1] = m.c[2][2] = m.c[3][3] = 1.0f;
	return m;
}

// Matrix product in the usual mathematical sense: (a * b) applied to a vector
// means "first b, then a".
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.c[k][i] * b.c[j][k];
			}
			r.c[j][i] = sum;
		}
	}
	return r;
}

// --- Affine transformations -------------------------------------------------

inline Mat4 translation(Vec3 t) {
	Mat4 m = identity();
	m.c[3][0] = t.x;
	m.c[3][1] = t.y;
	m.c[3][2] = t.z;
	return m;
}

inline Mat4 scaling(Vec3 s) {
	Mat4 m = identity();
	m.c[0][0] = s.x;
	m.c[1][1] = s.y;
	m.c[2][2] = s.z;
	return m;
}

inline Mat4 rotationX(float angle) {
	const float s = std::sin(angle);
	const float c = std::cos(angle);
	Mat4 m = identity();
	m.c[1][1] =  c;
	m.c[2][1] = -s;
	m.c[1][2] =  s;
	m.c[2][2] =  c;
	return m;
}

inline Mat4 rotationY(float angle) {
	const float s = std::sin(angle);
	const float c = std::cos(angle);
	Mat4 m = identity();
	m.c[0][0] =  c;
	m.c[2][0] =  s;
	m.c[0][2] = -s;
	m.c[2][2] =  c;
	return m;
}

inline Mat4 rotationZ(float angle) {
	const float s = std::sin(angle);
	const float c = std::cos(angle);
	Mat4 m = identity();
	m.c[0][0] =  c;
	m.c[1][0] = -s;
	m.c[0][1] =  s;
	m.c[1][1] =  c;
	return m;
}

// Матрица для нормалей: обратная транспонированная к блоку 3x3 матрицы модели.
// Если фигуру растянуть неравномерно, то нормали, умноженные на саму model,
// перестают быть перпендикулярны поверхности, а на эту матрицу -- остаются.
inline Mat4 normalMatrix(const Mat4& m) {
	const float a = m.c[0][0], b = m.c[1][0], c = m.c[2][0];
	const float d = m.c[0][1], e = m.c[1][1], f = m.c[2][1];
	const float g = m.c[0][2], h = m.c[1][2], i = m.c[2][2];

	// Матрица алгебраических дополнений. Обратная матрица равна её
	// транспонированной, поделённой на определитель, а нам нужна обратная
	// транспонированная -- то есть сама матрица дополнений, делённая на него.
	const float cofactors[3][3] = {
		{   e * i - f * h, -(d * i - f * g),   d * h - e * g},
		{-(b * i - c * h),   a * i - c * g, -(a * h - b * g)},
		{   b * f - c * e, -(a * f - c * d),   a * e - b * d},
	};

	const float det = a * cofactors[0][0] + b * cofactors[0][1] + c * cofactors[0][2];

	Mat4 r = identity();
	if (det == 0.0f) {
		return r; // вырожденный масштаб (ноль по какой-то оси)
	}

	for (int row = 0; row < 3; ++row) {
		for (int column = 0; column < 3; ++column) {
			r.c[column][row] = cofactors[row][column] / det;
		}
	}
	return r;
}

// Поворот, который переводит ось Z в заданное направление. Нужен, чтобы
// поставить конус прожектора вдоль направления его свечения.
inline Mat4 orientation(Vec3 direction) {
	const Vec3 forward = normalize(direction);

	// Опорный вектор не должен быть параллелен направлению.
	const Vec3 reference = (std::abs(forward.y) > 0.99f) ? Vec3{1.0f, 0.0f, 0.0f}
	                                                     : Vec3{0.0f, 1.0f, 0.0f};
	const Vec3 right = normalize(cross(reference, forward));
	const Vec3 up = cross(forward, right);

	Mat4 m = identity();
	m.c[0][0] = right.x;   m.c[0][1] = right.y;   m.c[0][2] = right.z;
	m.c[1][0] = up.x;      m.c[1][1] = up.y;      m.c[1][2] = up.z;
	m.c[2][0] = forward.x; m.c[2][1] = forward.y; m.c[2][2] = forward.z;
	return m;
}

// --- Camera and projections -------------------------------------------------

// Right-handed view matrix: the camera sits in `eye` and looks at `target`.
inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
	const Vec3 f = normalize(target - eye); // forward
	const Vec3 s = normalize(cross(f, up)); // right
	const Vec3 u = cross(s, f);             // true up

	Mat4 m = identity();
	m.c[0][0] =  s.x; m.c[1][0] =  s.y; m.c[2][0] =  s.z; m.c[3][0] = -dot(s, eye);
	m.c[0][1] =  u.x; m.c[1][1] =  u.y; m.c[2][1] =  u.z; m.c[3][1] = -dot(u, eye);
	m.c[0][2] = -f.x; m.c[1][2] = -f.y; m.c[2][2] = -f.z; m.c[3][2] =  dot(f, eye);
	return m;
}

// Both projections below are written for Vulkan's clip space:
//   * depth is mapped to [0, 1] (not [-1, 1] like in OpenGL);
//   * the Y axis of normalized device coordinates points DOWN, so the second
//     row is negated once here instead of flipping the viewport every frame.
//   * the view space is right-handed, the camera looks along -Z.

inline Mat4 perspective(float fovy, float aspect, float near, float far) {
	const float f = 1.0f / std::tan(fovy * 0.5f);

	Mat4 m;
	m.c[0][0] =  f / aspect;
	m.c[1][1] = -f;
	m.c[2][2] =  far / (near - far);
	m.c[2][3] = -1.0f;                      // w_clip = -z_view
	m.c[3][2] =  (near * far) / (near - far);
	return m;
}

inline Mat4 orthographic(float half_width, float half_height, float near, float far) {
	Mat4 m = identity();
	m.c[0][0] =  1.0f / half_width;
	m.c[1][1] = -1.0f / half_height;
	m.c[2][2] = -1.0f / (far - near);
	m.c[3][2] = -near / (far - near);
	return m;
}

} // namespace linalg
