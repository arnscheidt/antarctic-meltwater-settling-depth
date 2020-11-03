# basic code for the 3D large-eddy simulations of a line plume

# imports - note that the Project.toml file needs to be present for these to work correctly
using DelimitedFiles, Printf, Interpolations
using Plots
using CuArrays, CUDAnative, CUDAdrv

using Oceananigans
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations
using Oceananigans.Utils
using Oceananigans: Face, Cell

#####
##### Preliminaries
#####

# some constants
const km = 1000
const Ω_Earth = 7.292115e-5  # [s⁻¹]
const φ = -75  # degrees latitude

# architecture
# modify as appropriate: if no GPU, set arch = CPU() and delete the following line
arch = GPU()
device!(CuDevice(2))

FT = Float64

# domain details (grid cells, then dimensions in m)
Nx = 512 
Ny = 512
Nz = 96 

Lx = 5km
Ly = 5km
Lz = 400 

end_time = 6hour 

g = 9.81
β = 7.8*10^(-4)

# SET UP INITIAL CONDITION
# Option 1: Make up salinity and temperature profiles based on prescribed N
#
#Nb = 0.003
#ΔS = (Lz*Nb^2)/(g*β)
#zC = collect(((-Lz:Lz/Nz:0).+Lz/(2*Nz))[1:end-1])
#T₀ = 1*ones(Nz)
#S₀ = reverse(collect(34.6-ΔS:ΔS/(Nz-1):34.6))

# Option 2: Read T and S data from file
zC = collect(((-Lz:Lz/Nz:0).+Lz/(2*Nz))[1:end-1])
zdata = (collect(1:1:5400))
T = (readdlm("pig_data/pig_outflow_2014_T.txt")[:])
S = (readdlm("pig_data/pig_outflow_2014_S.txt")[:])

# Get the indices of all the non-NaN values.
T_good_inds = findall(!isnan, T)
S_good_inds = findall(!isnan, S)

# Create T and S arrays that do not contain NaNs, along with corresponding
# z values.
T_good = T[T_good_inds]
S_good = S[S_good_inds]

z_T = zdata[T_good_inds]
z_S = zdata[S_good_inds]

# Linearly interpolate T and S profiles to model grid.
Ti = LinearInterpolation(z_T, T_good, extrapolation_bc=Interpolations.Flat())
Si = LinearInterpolation(z_S, S_good, extrapolation_bc=Interpolations.Flat())

T₀ = Ti.(-zC)
S₀ = Si.(-zC)

# end initial condition setup

# convert to CuArray if we are running on a GPU
if arch == GPU()
    T₀ = CuArray(T₀)
    S₀ = CuArray(S₀)
end

# Meltwater source location - implemented as a box
# width in terms of grid cells 
widthx = 100 
widthy = 10
source_corners = ((256-widthx/2,0,0),(256+widthx/2,widthy,1))

# specify the integrated buoyancy source (m^4/s^3) (i.e. F)
# then calculate salinity fluxes into bottom boxes
B_flux = 1 
A = (Ly/Ny)*(source_corners[2][2] - source_corners[1][2])*(Lx/Nx)*(source_corners[2][1] - source_corners[1][1])
S_flux = -B_flux/(A*g*β*(Lz/Nz))

# meltwater passive tracer flux
m_flux = 1

# Forcing functions 
@inline T_forcing(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse(p.source_corners[1][1]<=i<=p.source_corners[2][1] && p.source_corners[1][2]<=j<=p.source_corners[2][2] && p.source_corners[1][3]<=k<=p.source_corners[2][3], 0, 0) 
              
@inline S_forcing(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse(p.source_corners[1][1]<=i<=p.source_corners[2][1] && p.source_corners[1][2]<=j<=p.source_corners[2][2] && p.source_corners[1][3]<=k<=p.source_corners[2][3], p.S_flux, 0) 

@inline m_relax(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse(p.source_corners[1][1]<=i<=p.source_corners[2][1] && p.source_corners[1][2]<=j<=p.source_corners[2][2] && p.source_corners[1][3]<=k<=p.source_corners[2][3], p.m_flux, 0) 

params = (source_corners=source_corners, S_flux=S_flux, m_flux = m_flux, T₀=T₀,S₀=S₀)

forcing = ModelForcing(T = T_forcing, S = S_forcing, meltwater = m_relax)

#####
##### Set up model and simulation
#####

topology = (Oceananigans.Periodic, Bounded, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), x=(-Lx/2, Lx/2), y=(0, Ly), z=(-Lz, 0))

# equation of state - pick one!
# eos = LinearEquationOfState()
eos = RoquetIdealizedNonlinearEquationOfState(:freezing)

model = IncompressibleModel(
           architecture = arch,
             float_type = FT,
                   grid = grid,
                tracers = (:T, :S, :meltwater),
               coriolis = FPlane(rotation_rate=Ω_Earth, latitude=φ),
               buoyancy = SeawaterBuoyancy(equation_of_state=eos),
                closure = AnisotropicMinimumDissipation(),
                forcing = forcing,
             parameters = params
)

#####
##### Setting up initial conditions
#####

T₀_3D = repeat(reshape(T₀, 1, 1, Nz), Nx, Ny, 1)
S₀_3D = repeat(reshape(S₀, 1, 1, Nz), Nx, Ny, 1)

set!(model.tracers.T, T₀_3D)
set!(model.tracers.S, S₀_3D)

#####
##### Write out 3D fields and slices to NetCDF files.
#####

# Define vorticity computation
u, v, w = model.velocities
vorticity_operation = ∂x(v) - ∂y(u)
ω = Field(Face, Face, Cell, model.architecture, model.grid, TracerBoundaryConditions(grid))
vorticity_computation = Computation(vorticity_operation, ω)

function get_vorticity(model)
    compute!(vorticity_computation)
    return Array(interior(ω))
end

fields = Dict(
        "u" => model.velocities.u,
        "v" => model.velocities.v,
        "w" => model.velocities.w,
        "T" => model.tracers.T,
        "S" => model.tracers.S,
"meltwater" => model.tracers.meltwater,
       "nu" => model.diffusivities.νₑ,
   "kappaT" => model.diffusivities.κₑ.T,
   "kappaS" => model.diffusivities.κₑ.S,
   "vorticity" => get_vorticity
)

dimensions = Dict(
    "vorticity" => ("xF", "yF", "zC")
)

output_attributes = Dict(
"meltwater" => Dict("longname" => "Meltwater concentration"),
       "nu" => Dict("longname" => "Nonlinear LES viscosity", "units" => "m^2/s"),
   "kappaT" => Dict("longname" => "Nonlinear LES diffusivity for temperature", "units" => "m^2/s"),
   "kappaS" => Dict("longname" => "Nonlinear LES diffusivity for salinity", "units" => "m^2/s"),
   "vorticity" => Dict("longname" => "Vorticity", "units" => "1/s")
)

eos_name(::LinearEquationOfState) = "LinearEOS"
eos_name(::RoquetIdealizedNonlinearEquationOfState) = "RoquetEOS"
prefix = "$(Nx)x$(Ny)x$(Nz)_F$(B_flux)_w$(widthx)x$(widthy)c"

#####
##### Print banner
#####

@printf("""

    Simulating ocean dynamics of meltwater outflow from beneath Antarctic ice shelves
        N : %d, %d, %d
        L : %.3g, %.3g, %.3g [km]
        Δ : %.3g, %.3g, %.3g [m]
        φ : %.3g [latitude]
        f : %.3e [s⁻¹]
     days : %d
        F : %.2f [m⁴s⁻³]
  closure : %s
      EoS : %s

""", model.grid.Nx, model.grid.Ny, model.grid.Nz,
     model.grid.Lx / km, model.grid.Ly / km, model.grid.Lz / km,
     model.grid.Δx, model.grid.Δy, model.grid.Δz,
     φ, model.coriolis.f, end_time / day,
     B_flux, 
     typeof(model.closure), typeof(model.buoyancy.equation_of_state))

#####
##### Time step!
#####

# Wizard utility that calculates safe adaptive time steps.
wizard = TimeStepWizard(cfl=0.2, Δt=1second, max_change=1.2, max_Δt=10second)

# Number of time steps to perform at a time before printing a progress
# statement and updating the adaptive time step.
Ni = 20

# CFL utilities for reporting stability criterions.
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)

function progress_statement(simulation)
    model = simulation.model
    C_mw = model.tracers.meltwater  # Convenient alias
    
    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / end_time)

    # Find maximum velocities.
    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    # Find maximum ν and κ.
    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)

    # Print progress statement.
    i, t = model.clock.iteration, model.clock.time
    @printf("[%06.2f%%] i: %d, t: %5.2f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, νκmax: (%6.3g, %6.3g), νκCFL: %6.4g, next Δt: %8.5g s\n",
            progress, i, t / day, umax, vmax, wmax, cfl(model), νmax, κmax, dcfl(model), wizard.Δt)
end

_ω = get_vorticity(model)
#display(_ω)
#@show typeof(_ω)

# Simulation that manages time stepping.
simulation = Simulation(model, Δt=wizard, stop_time=end_time, progress=progress_statement, progress_frequency=Ni)

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, filename = prefix * "fields.nc",
                       interval = hour, output_attributes = output_attributes,
		       dimensions = dimensions)

fields_without_vorticity = copy(fields)
delete!(fields_without_vorticity,"vorticity")

simulation.output_writers[:along_channel_slice] =
    NetCDFOutputWriter(model, fields_without_vorticity, 
		       filename = prefix * "yz.nc",
                       interval = 5minute, output_attributes = output_attributes,
		       dimensions = dimensions, xC = Int(Nx/2), xF = Int(Nx/2))

simulation.output_writers[:along_front_slice] =
    NetCDFOutputWriter(model, fields_without_vorticity, 
		       filename = prefix * "xz.nc",
                       interval = 5minute, output_attributes = output_attributes,
                       dimensions = dimensions, yC = 1, yF = 1)

run!(simulation)

for ow in simulation.output_writers
     ow isa NetCDFOutputWriter && close(ow)
end
