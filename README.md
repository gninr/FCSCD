# FCSCD
Force computation using shape calculus on dielectrics

## Requirements
* Requires Eigen3, CMake, GNU compiler.
* If Eigen 3 unavailable is, use "sudo apt-get install -y libeigen3-dev".
* Get the submodule files by "git submodule update --init --recursive".

## Using CMake
* Create a "build" directory in the project directory. Use "cmake .." from inside the build directory. It is recommended to specify the macro "-DCMAKE_BUILD_TYPE=Release" to compile in release mode for faster runs.

## Generating square-kite meshes
* From the build folder, execute "make geom" to compile square-kite geometry generator.
* Run "./geom level" to generate the geometry with desired level of refinement.
* Open the generated geometry files in Gmsh to build meshes. Save meshes to build/examples/meshes folder.

## Building targets
* From the build folder, execute "make target_name" to compile target_name.
* All the compiled executables lie in the folder build/examples which can be run by "./target_name eps_2". eps_2 is used to set the relative permittivity since eps_1 = 1 by default.
* Use "python plot.py shape_name eps_2" to visualize the result. (shape_name = square/kite)

## Target names for net force
* For square-shaped inner domain using BEM: square_bem
* For square-shaped inner domain using FEM: square_fem
* For kite-shaped inner domain using BEM: kite_bem
* For kite_shaped inner domain using FEM: kite_fem