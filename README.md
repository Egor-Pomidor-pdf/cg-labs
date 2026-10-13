# 🌋 Vulkan Starter App

## Getting started

You need C++ compiler, Vulkan SDK and CMake installed before you can build this project.

This project uses C++20 standard and thus requires either of those compilers:
- GCC 10.X
- Clang 10
- Microsoft Visual Studio 2019

This is officially tested on *Windows* and *GNU/Linux platforms*, no *macOS* support yet.
If you have a working macOS solution of this code, consider submitting a PR so others
can build this example code without a hassle!

<ins>**1. Downloading the repository**</ins>

Start by cloning the repository with `git clone --depth 1 https://github.com/vladeemerr/vulkan-starter-app`

This repository does not contain any submodules, it utilizes CMake's `FetchContent` feature instead.

<ins>**2. Configuring the project**</ins>

Run either one of the CMake lines to download dependencies and configure the project:

```bash
cmake --preset debug       # for GNU/Linux (GCC/Clang)
cmake --preset msvc-debug  # for Windows (Visual Studio 2019)
cmake --preset mingw-debug # for Windows (MinGW)
```

If you wish to build in `release` mode, change `debug` to `release`.

If changes are made (added/removed files), or if you want to regenerate project files, rerun the command above.

<ins>**3. Building**</ins>

To build the project, use the line below. You are most likely using `debug` preset, so
the directory that will eventually contain your build files is named `build-debug`.

Likewise for `release` that directory will be named `build-release`

Run one those commands, depending on which preset you chose:

```bash
cmake --build build-debug --parallel # for debug
cmake --build build-release --parallel # for release
```

### Running

`build-*` directory will contain the executable in one of the subdirectories after successful build.

For `msvc-{debug|release}` builds output subdirectory is set to `Debug` or `Release` respectively.
For other configurations output subdirectory is set to `vulkan-starter-app`.

**Make sure your working directory is set to the project root!**
Project root is where this README file resides. Otherwise, the
code responsible for loading shaders or other resources from files will fail,
because relative paths are used.

### Compiling shaders

`CMakeLists.txt` has a build recipe for compiling shader files
along with an application. Look for a comment in this file to see
how to compile your shaders.

---

# Лабораторная работа №1 — Основы 3D-графики

**Вариант 11: правильный икосаэдр.**

Выполнена базовая часть задания и первые пять дополнительных (на оценку 4):
построение 3D-объекта, проекция на 2D-плоскость, работа с матрицами
перспективной и ортографической проекции и аффинными преобразованиями в
реальном времени, движение по сложной траектории и управление цветом фигуры
из интерфейса.

## Что сделано

* Правильный икосаэдр: 12 вершин, 30 рёбер, 20 треугольных граней.
  Координаты вершин — это вершины трёх взаимно перпендикулярных «золотых»
  прямоугольников со сторонами `2 × 2φ`, где `φ = (1 + √5) / 2`.
  Все рёбра получаются равными `2`, после нормировки фигура вписана в единичную
  сферу.
* Каждой грани в вершинном буфере соответствуют свои три вершины (20 × 3 = 60),
  чтобы соседние грани могли задать разный цвет в общей точке. Цвет вершины
  назначается процедурно — по её позиции в локальной системе координат фигуры,
  с добавкой направления на центр грани; в шейдере он умножается на цвет,
  выбранный в интерфейсе.
* Отрисовка в реальном времени с тестом глубины (`VK_COMPARE_OP_LESS`).
* Матрица `MVP = P · V · M` и выбранный цвет пересчитываются каждый кадр на CPU
  и передаются в шейдеры через push-константы (80 байт из гарантированных 128).
* Движение по замкнутой кривой Лиссажу: горизонтальная окружность и колебание
  по вертикали, число лепестков которого задаётся в интерфейсе. Фигура может
  доворачиваться по направлению движения — оно берётся как разность двух
  близких точек траектории.
* Интерфейс на ImGUI позволяет на лету:
  * переключать проекцию — перспективная / ортографическая;
  * менять угол обзора (или размер области видимости) и расстояние до камеры;
  * задавать аффинные преобразования фигуры — перенос, поворот вокруг трёх
    осей и растяжение;
  * включать и выключать автоматическое вращение и менять его скорость;
  * запускать, ставить на паузу и сбрасывать движение по траектории, менять его
    скорость, радиус, амплитуды по вертикали и глубине и число лепестков кривой;
  * выбирать цвет фигуры (`ColorEdit`), на который умножается процедурный цвет
    вершин.

## Структура кода

| Файл | Назначение |
| --- | --- |
| `source/linalg.hpp` | Вектора и матрицы 4×4: аффинные преобразования, `lookAt`, перспективная и ортографическая проекции |
| `source/graphics.hpp/.cpp` | Обёртки над повторяющимся кодом Vulkan: буферы (VMA) и загрузка шейдерных модулей |
| `source/application.cpp` | Геометрия икосаэдра, графический конвейер, интерфейс и запись команд отрисовки |
| `shaders/object.vert`, `shaders/object.frag` | Вершинный и фрагментный шейдеры |

Файлы `source/main.cpp` и `source/graphics_internal.*` оставлены без изменений
относительно стартового кода.

## Замечания по реализации

* Матрицы проекции написаны сразу под систему координат Vulkan: глубина
  отображается в диапазон `[0, 1]`, ось Y направлена вниз (поэтому вторая
  строка матрицы проекции домножена на −1).
* Отсечение задних граней выключено (`VK_CULL_MODE_NONE`): фигуру можно
  растянуть отрицательным масштабом, что меняет направление обхода
  треугольников, а видимость выпуклого тела корректно решает буфер глубины.
* Вершинный буфер создаётся в памяти, видимой и CPU, и GPU. Для такого объёма
  геометрии этого достаточно; для больших моделей потребовался бы
  промежуточный (staging) буфер и копирование в память устройства.
* Подписи в интерфейсе на латинице: шрифт ImGUI по умолчанию не содержит
  глифов кириллицы.

## Сборка и запуск

```bash
cmake --preset debug
cmake --build build-debug --parallel
./build-debug/vulkan-starter-app
```

Рабочий каталог при запуске должен быть корнем проекта — шейдеры загружаются
по относительному пути `shaders/*.spv`. Шейдеры компилируются автоматически
через `glslc` на этапе сборки.
