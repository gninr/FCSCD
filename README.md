# FCSCD
Force computation using shape calculus on dielectrics

## Requirements
* Requires Eigen3, CMake, GNU compiler.
* If Eigen 3 unavailable is, use "sudo apt-get install -y libeigen3-dev".
* Get the submodule files by "git submodule update --init --recursive".

## Using CMake
* Create a "build" directory in the project directory. Use "cmake .." from inside the build directory. It is recommended to specify the macro "-DCMAKE_BUILD_TYPE=Release" to compile in release mode for faster runs.

## Building targets for shape derivative
* From the build folder, execute "make target_name" to compile target_name.
* All the compiled executables lie in the folder build/examples.
* Use "python plot.py shape_name" to visualize the result. (shape_name = square/kite)

## Target names for net force
* For square-shaped D using BEM: square_bem
* For square-shaped D using FEM: square_fem
* For kite-shaped D using BEM: kite_bem
* For kite_shaped D using FEM: kite_fem