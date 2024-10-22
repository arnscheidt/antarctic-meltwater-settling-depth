# antarctic-meltwater-settling-depth
Code to set up the simulations in Arnscheidt et al. (2021) "On the settling depth of meltwater escaping from underneath an Antarctic ice shelf". Each model runs from a single script, and the relevant parameter values can be straightforwardly modified within that script.

The simple line plume model can be run from `1d_line_plume.py` with a working [Python3](https://www.python.org/downloads/) environment.

The large-eddy simulations run in [Julia](https://julialang.org/) using [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). Ideally, the more computationally intensive simulations should be run using Graphical Processing Units (GPUs). 2-D meltwater plume large-eddy simulations can be run from `les_2d_constant.jl` and `les_2d_restoring.jl`, which use a constant buoyancy source and a restoring buoyancy source respectively. 3-D large-eddy simulations can be run from `les_3d.jl`. These rely on a specific version of Oceananigans.jl and thus require the Project.toml file: they can be run using `julia --project` and then typing `include("filename.jl")`. 

`pig_data` contains temperature and salinity profiles from near the meltwater outflow from beneath Pine Island glacier in 2009 and 2014; these can be used as initial conditions in `les_3d.jl`.
