ENV["PS_PACKAGE"] = :Threads

using JustPIC
using MAT, CUDA
# using MuladdMacro, MAT, CUDA
# using ParallelStencil
# using GLMakie
# using StencilInterpolations
# import StencilInterpolations: _grid2particle, parent_cell, isinside

push!(LOAD_PATH, "..")
PS_PACKAGE = Symbol(ENV["PS_PACKAGE"])

include("../src/data.jl")

## -------------------------------------------------------------

function random_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny)
    ncells = (nx - 1) * (ny - 1)
    np = max_xcell * ncells
    px, py = ntuple(_ -> zeros(Float64, np), Val(2))

    # index = zeros(UInt32, np)
    index = falses(np)
    inject = falses(ncells)
    for j in 1:(ny - 1), i in 1:(nx - 1)
        # lowermost-left corner of the cell
        x0, y0 = x[i], y[j]
        # cell index
        cell = i + (nx - 1) * (j - 1)
        for l in 1:max_xcell
            if l ≤ nxcell
                idx = (cell - 1) * max_xcell + l # particle index
                px[idx] = rand() * dx + x0
                py[idx] = rand() * dy + y0
                index[l] = true
                # index[idx] = idx
            end
        end
    end

    return Particles((px, py), index, inject, nxcell, max_xcell, np)
end

function twoxtwo_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny, lx, ly)
    rad2 = 2.0
    ncells = (nx - 1) * (ny - 1)
    np = max_xcell * ncells
    px, py, pT = ntuple(_ -> fill(NaN, np), Val(3))

    min_xcell = ceil(Int, nxcell / 2)

    # index = zeros(UInt32, np)
    inject = falses(ncells)
    index = falses(np)
    @inbounds for j in 1:(ny - 1), i in 1:(nx - 1)
        # lowermost-left corner of the cell
        x0, y0 = x[i], y[j]
        # cell index
        cell = i + (nx - 1) * (j - 1)
        # index of first particle in cell
        idx = (cell - 1) * max_xcell + 1
        # add 4 new particles in a 2x2 manner + some small random perturbation
        px[idx] = x0 + dx * (1 / 3) * (1.0 + (rand() - 0.5))
        px[idx + 1] = x0 + dx * (2 / 3) * (1.0 + (rand() - 0.5))
        px[idx + 2] = x0 + dx * (1 / 3) * (1.0 + (rand() - 0.5))
        px[idx + 3] = x0 + dx * (2 / 3) * (1.0 + (rand() - 0.5))
        py[idx] = y0 + dy * (1 / 3) * (1.0 + (rand() - 0.5))
        py[idx + 1] = y0 + dy * (1 / 3) * (1.0 + (rand() - 0.5))
        py[idx + 2] = y0 + dy * (2 / 3) * (1.0 + (rand() - 0.5))
        py[idx + 3] = y0 + dy * (2 / 3) * (1.0 + (rand() - 0.5))
        # fill index array
        for l in idx:(idx + nxcell - 1)
            # index[l] = l
            index[l] = true
            pT[l] = exp(
                -((x0 + px[l] * dx - lx / 2)^2 + (y0 + py[l] * dy - ly / 2)^2) / rad2
            )
        end
    end

    if PS_PACKAGE === :CUDA
        pxi = CuArray.((px, py))
        return Particles(
            pxi, CuArray(index), CuArray(inject), nxcell, max_xcell, min_xcell, np, (nx, ny)
        ),
        CuArray(pT)

    else
        return Particles((px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)), pT
    end
end

function main(Vx, Vy; nx=42, ny=42, nxcell=4, α=2 / 3, nt=1_000, viz=false)
    V = (Vx, Vy)
    vx0, vy0 = maximum(abs.(Vx)), maximum(abs.(Vy))

    # model domain
    lx = ly = 10
    dx, dy = lx / (nx - 1), ly / (ny - 1)
    dxi = (dx, dy)
    nxi = (nx, ny)
    x = LinRange(0, lx, nx)
    y = LinRange(0, ly, ny)
    grid = (x, y)

    T = PS_PACKAGE === :CUDA ? CUDA.zeros(Float64, nx, ny) : zeros(nx, ny)
    T0 = similar(T)

    # random particles
    max_xcell = nxcell * 2
    # particles = random_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny)
    particles, pT = twoxtwo_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny, lx, ly)

    # field to interpolate
    dt = min(dx, dy) / max(abs(vx0), abs(vy0)) * 0.25
    it = 1
    t = 0.0
    nsave = 10
    injected_cells = Vector{Int64}(undef, nt)
    it_time = Vector{Float64}(undef, nt)

    args = (pT,)
    
    while it ≤ nt
        if it == nt ÷ 2
            dt = -dt
        end

        t1 = @elapsed begin
            gathering!(T, pT, grid, particles.coords, particles.upper_buffer, particles.lower_buffer)

            # advect particles in space
            advection_RK2!(particles, V, grid, dxi, dt, α)

            # advect particles in memory
            shuffle_particles!(particles, grid, dxi, nxi, args)

            injected_cells[it] = sum(particles.inject)

            check_injection(particles.inject) && (
                inject_particles!(particles, grid, nxi, dxi);
                grid2particle!(pT, grid, T, particles.coords)
            )
        end

        it_time[it] = t1
        it += 1

        viz && (it % nsave == 0) && plot(x, y, T, particles, pT, it)

    end

    return injected_cells, it_time
end

nx = ny = 42
nxcell = 4
nt = 1000
α = 2 / 3

Vx, Vy = load_benchmark_data("data/data41_benchmark.mat")
# Vx, Vy = CuArray(Vx), CuArray(Vy)

injected_23, t_23 = main(Vx, Vy; nx=nx, ny=ny, α=2 / 3, nt=1000, viz = false)
injected_rk2, t_rk2 = main(Vx, Vy; nx=nx, ny=ny, α=0.5, nt=1000)
injected_heun, t_heun = main(Vx, Vy; nx=nx, ny=ny, α=1.0, nt=1000)

# @btime main($Vx, $Vy; α = $2/3, nt=$1000) # CPU: 1.202 s (156122 allocations: 109.18 MiB)
# @btime main($Vx, $Vy; nx=$40, ny=$40, α = $2/3, nt=$1000) # GPU: 0.445890 s (271.84 k allocations: 18.003 MiB)

lines(cumsum(injected_rk2); color=:red)
lines!(cumsum(injected_heun); color=:green)
lines!(cumsum(injected_23); color=:blue)

lines(cumsum(t_rk2); color=:red)
lines!(cumsum(t_heun); color=:green)
lines!(cumsum(t_23); color=:blue)

df = DataFrame(
    injected_23 = injected_23,
    injected_rk2 = injected_rk2,
    injected_heun = injected_heun,
    t_23 = t_23,
    t_rk2 = t_rk2,
    t_heun = t_heun,
)

CSV.write("CPU_baseline.csv", df)