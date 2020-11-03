using DelimitedFiles, Printf
using Plots
using CuArrays

using Oceananigans
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations
using Oceananigans.Utils

using Oceananigans: Face, Cell

# setting up a 2-d model in the style of NG17

#####
##### Some useful constants
#####

const km = 1000
const Ω_Earth =  7.292115e-5  # [s⁻¹]
const φ = -75  # degrees latitude

#####
##### Model grid and domain size
#####

arch = CPU()

FT = Float64

Nx = 1
Ny = 512 
Nz = 96 

Lx = 5km/Ny
Ly = 5km
Lz = 300 

end_time = 1day 

# Make up temperature profiles
zC = collect(((-Lz:Lz/Nz:0).+Lz/(2*Nz))[1:end-1])
T₀ = collect(1:2/(Nz-1):3)
S₀ = 34*ones(Nz) 

# convert to CuArray if we are running on a GPU
if arch == GPU()
    T₀ = CuArray(T₀)
    S₀ = CuArray(S₀)
end

#####
##### Set up relaxation areas for the meltwater source and for the northern boundary 
#####

# Meltwater source location - implemented as a box
source_corners_m = ((0,0,0),(Lx,100,1))
N = (Nx,Ny,Nz)
L = (Lx,Ly,Lz)
source_corners = (Int.(ceil.(source_corners_m[1].*N./L)),Int.(ceil.(source_corners_m[2].*N./L)))

λ = 1/(20)  # Relaxation timescale [s⁻¹].

# specify the integrated buoyancy source (m^4/s^3)
# then calculate temperature fluxes into bottom boxes
B_flux = 0.007 
α = 1.67 * 10^(-4)
g = 9.81
A = (source_corners_m[2][2] - source_corners_m[2][1])*Lx
T_flux = B_flux/(A*g*α*(Lz/Nz))

# Specify width of stable relaxation area
stable_relaxation_width_m = 400.0
stable_relaxation_width = Int(ceil(stable_relaxation_width_m.*Ny./Ly))

# Forcing functions 
@inline T_relax(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse(p.source_corners[1][1]<=i<=p.source_corners[2][1] && p.source_corners[1][2]<=j<=p.source_corners[2][2] && p.source_corners[1][3]<=k<=p.source_corners[2][3], p.T_flux, 0) +
              ifelse(j>grid.Ny-p.stable_relaxation_width, -p.λ * (C.T[i, j, k] - p.T₀[k]), 0)

@inline S_relax(i, j, k, grid, time, U, C, p) =
    @inbounds ifelse(p.source_corners[1][1]<=i<=p.source_corners[2][1] && p.source_corners[1][2]<=j<=p.source_corners[2][2] && p.source_corners[1][3]<=k<=p.source_corners[2][3], 0, 0) +
              ifelse(j>grid.Ny-p.stable_relaxation_width, -p.λ * (C.S[i, j, k] - p.S₀[k]), 0)

params = (source_corners=source_corners, T_flux=T_flux, λ=λ, stable_relaxation_width=stable_relaxation_width, T₀=T₀,S₀=S₀)

forcing = ModelForcing(T = T_relax, S = S_relax)

#####
##### Set up model and simulation
#####

topology = (Periodic, Bounded, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), x=(-Lx/2, Lx/2), y=(0, Ly), z=(-Lz, 0))

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

# Set meltwater concentration to 1 at the source.
model.tracers.meltwater.data[source_corners[1][1]:source_corners[2][1],source_corners[1][2]:source_corners[2][2],source_corners[1][3]:source_corners[2][3]] .= 1  

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
bfstr = "$(B_flux)"
prefix = "$(Nx)x$(Ny)x$(Nz)_b"bfstr[1]*bfstr[3:end]*"_"

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
        B : %.2f [m⁴s⁻³]
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
wizard = TimeStepWizard(cfl=0.1, Δt=0.1second, max_change=1.2, max_Δt=10second)

# Number of time steps to perform at a time before printing a progress
# statement and updating the adaptive time step.
Ni = 20

# CFL utilities for reporting stability criterions.
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)

function progress_statement(simulation)
    model = simulation.model
    C_mw = model.tracers.meltwater  # Convenient alias
    
    # add passive meltwater tracer at source, remove at boundary
    C_mw.data[source_corners[1][1]:source_corners[2][1],source_corners[1][2]:source_corners[2][2],source_corners[1][3]:source_corners[2][3]] .= 1
    C_mw.data[:,Ny-stable_relaxation_width:Ny,:] .= 0

    ## Normalize meltwater concentration to be 0 <= C_mw <= 1.
    #C_mw.data .= max.(0, C_mw.data)
    #C_m w.data .= C_mw.data ./ maximum(C_mw.data)
    
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
# display(_ω)
# @show typeof(_ω)

# Simulation that manages time stepping.
simulation = Simulation(model, Δt=wizard, stop_time=end_time, progress=progress_statement, progress_frequency=Ni)

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, filename = prefix * "fields.nc",
                       interval = 6hour, output_attributes = output_attributes,
		       dimensions = dimensions)

simulation.output_writers[:along_channel_slice] =
    NetCDFOutputWriter(model, fields, filename = prefix * "along_channel_yz_slice.nc",
                       interval = 5minute, output_attributes = output_attributes,
                       dimensions = dimensions, xC = 1, xF = 1)

run!(simulation)

for ow in simulation.output_writers
    ow isa NetCDFOutputWriter && close(ow)
end
