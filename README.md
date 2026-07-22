# Initial Project Setup (Run Once)
Configure the CMake build directory and generate the LSP compile commands:

```Bash
cmake -B build
```
Symlink the generated compile commands to the project root clangd LSP detects CUDA flags:

```Bash
ln -s build/compile_commands.json .
```
# Build & Run Workflow
Compile the project across all available CPU cores:

```Bash
cmake -B build -S .
cmake --build build --target matrix_operations
```
Execute the compiled CUDA binary:

```Bash
./build/src/matrix-operations/matrix_operations
```
# High-Speed Builds with Ninja (Optional)
Install the Ninja build backend (Debian/Ubuntu):

```Bash
sudo apt install ninja-build
```

Install the Ninja build backend (Arch Linux):

```Bash
sudo pacman -S ninja
```
Configure CMake to use Ninja instead of standard Makefiles:

```Bash
cmake -B build -G Ninja
```
# Profiling & Benchmarking
Profile memory transfers and kernel execution times using NVIDIA Nsight Systems:

```Bash
nsys profile --stats=true ./build/main
```

```Bash
compute-sanitizer --tool=memcheck ./build/main
```
