ENV["PS_PACKAGE"] = :Threads
# using JustPIC
using MAT
# using ProfileCanvas
using CUDA
# using GLMakie

using MuladdMacro
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(Threads, Float64, 2)

# include(
#     "C:\\Users\\albert\\Desktop\\StencilInterpolations.jl\\src\\StencilInterpolations.jl"
# )
using StencilInterpolations

# import .StencilInterpolations: _grid2particle

push!(LOAD_PATH, "..")
PS_PACKAGE = Symbol(ENV["PS_PACKAGE"])

include("../src/particles.jl")
include("../src/utils.jl")
include("../src/advection.jl")
include("../src/injection.jl")
include("../src/shuffle.jl")
include("../src/staggered/centered.jl")
include("../src/staggered/velocity.jl")
include("../src/data.jl")

## -------------------------------------------------------------
function plot_particles(particles)
    ii = findall(particles.index .=== true)
    return scatter(particles.coords[1][ii], particles.coords[2][ii])
end

function plot_T(particles, pT)
    ii = findall(particles.index .=== true)
    return scatter(
        particles.coords[1][ii], particles.coords[2][ii]; color=pT[ii], colormap=:batlow
    )
end

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
        return Particles(
            (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
        ),
        pT
    end
end

function twoxtwo_particles2D(nxcell, max_xcell, x, y, dx, dy, nx, ny, lx, ly)
    rad2 = 2.0
    ncells = nx * ny
    np = max_xcell * ncells
    dx_2 = dx * 0.5
    dy_2 = dy * 0.5
    px, py, pT = ntuple(_ -> fill(NaN, max_xcell, nx, ny), Val(3))
    min_xcell = ceil(Int, nxcell / 2)
    # min_xcell = 4

    # index = zeros(UInt32, np)
    inject = falses(nx, ny)
    index = falses(max_xcell, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        # center of the cell
        x0, y0 = x[i], y[j]
        xv, yv = (i - 1) * dx, (j - 1) * dy
        # index of first particle in cell
        idx = 1
        # add 4 new particles in a 2x2 manner + some small random perturbation
        px[idx, i, j] = x0 - 0.25 * dx_2 # * (1.0 + 0.15*(rand() - 0.5))
        px[idx + 1, i, j] = x0 + 0.25 * dx_2 # * (1.0 + 0.15*(rand() - 0.5))
        px[idx + 2, i, j] = x0 - 0.25 * dx_2 # * (1.0 + 0.15*(rand() - 0.5))
        px[idx + 3, i, j] = x0 + 0.25 * dx_2 # * (1.0 + 0.15*(rand() - 0.5))
        py[idx, i, j] = y0 - 0.25 * dy_2 # * (1.0 + 0.15*(rand() - 0.5))
        py[idx + 1, i, j] = y0 - 0.25 * dy_2 # * (1.0 + 0.15*(rand() - 0.5))
        py[idx + 2, i, j] = y0 + 0.25 * dy_2 # * (1.0 + 0.15*(rand() - 0.5))
        py[idx + 3, i, j] = y0 + 0.25 * dy_2 # * (1.0 + 0.15*(rand() - 0.5))
        # fill index array
        for l in 1:nxcell
            # px[l, i, j] = x0 + dx/3*(1.0 + (rand() - 0.5))
            # py[l, i, j] = y0 + dy/3*(1.0 + (rand() - 0.5))

            index[l, i, j] = true
            pT[l, i, j] = exp(
                -(
                    (xv + px[l, i, j] * dx * 0.5 - lx / 2)^2 +
                    (yv + py[l, i, j] * dy * 0.5 - ly / 2)^2
                ) / rad2,
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
        return Particles(
            (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
        ),
        pT
    end
end

function main(Vx, Vy; nx=42, ny=42, nxcell=4, α=2 / 3, nt=1_000, viz=false)
    Vx = Vx[1:(end - 1), :]
    Vy = Vy[:, 1:(end - 1)]
    Vx = PS_PACKAGE === :CUDA ? CuArray(Vx) : Vx
    Vy = PS_PACKAGE === :CUDA ? CuArray(Vy) : Vy

    V = (Vx, Vy)
    vx0, vy0 = maximum(abs.(Vx)), maximum(abs.(Vy))

    # model domain
    lx = ly = 10.0
    dx, dy = lx / nx, ly / ny
    dxi = (dx, dy)
    nxi = (nx, ny)
    x = LinRange(dx / 2, lx - dx / 2, nx)
    y = LinRange(dy / 2, ly - dy / 2, ny)
    grid = (x, y)
    # velocity grids
    xv = 0:dx:lx
    yv = 0:dy:ly
    yvx = (-dx / 2):dy:(ly + dy / 2)
    xvy = (-dx / 2):dx:(lx + dx / 2)
    grid_vx = (xv, yvx)
    grid_vy = (xvy, yv)

    T = PS_PACKAGE === :CUDA ? CUDA.zeros(Float64, nx + 2, ny + 2) : zeros(nx + 2, ny + 2)
    T0 = similar(T)

    # random particles
    max_xcell = nxcell + 2
    # particles = random_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny)
    particles, pT = twoxtwo_particles2D(nxcell, max_xcell, x, y, dx, dy, nx, ny, lx, ly)

    # field to interpolate
    dt = min(dx, dy) / max(abs(vx0), abs(vy0)) * 0.25
    it = 1
    t = 0.0
    nsave = 2
    injected_cells = Vector{Int64}(undef, nt)
    it_time = Vector{Float64}(undef, nt)

    args = (pT,)

    while it ≤ nt
        if it == nt ÷ 2
            dt = -dt
        end

        t1 = @elapsed begin
            copyto!(T0, T)

            #
            # we would run diffusion here
            #

            for argsi in args
                # grid to particles
                # int2part!(argsi, T, T0, particles, grid)
                grid2particle_xcell!(argsi, grid, T, particles.coords)
            end

            # advect particles in space
            # advection_RK2!(particles, V, grid, dt, α)
            advection_RK2_edges!(particles, V, grid_vx, grid_vy, dt, α)

            # advect particles in memory
            shuffle_particles!(particles, grid, dxi, nxi, args)

            do_we_inject = check_injection(particles)
            injected_cells[it] = sum(particles.inject)

            if do_we_inject
                # println("Injecting $(sum(particles.inject))")
                inject_particles!(particles, args, (T, T0), grid)
            end

            for argsi in args
                # advected particles to grid
                gathering_xcell!(T, argsi, grid, particles.coords)
            end
        end

        it_time[it] = t1
        it += 1

        # viz && (it % nsave == 0) && plot(x, y, T, particles, pT, it)
    end

    return injected_cells, it_time
end

nx = ny = 40
nxcell = 4
nt = 1000
α = 2 / 3

Vx, Vy = load_benchmark_data("data/data41_benchmark.mat")

# injected_23, t_23 = main(Vx, Vy; nx=nx, ny=ny, α=2 / 3, nt=100, viz=false);
# @btime main($Vx, $Vy; nx=$nx, ny=$ny, α=$(2 / 3), nt=$1000, viz = $false);
# CUDA: 531.669 ms (407945 allocations: 32.66 MiB)

# injected_rk2, t_rk2 = main(Vx, Vy; nx=nx, ny=ny, α=0.5, nt=1000)
# injected_23, t_23 = main(Vx, Vy; nx=nx, ny=ny, α=2 / 3, nt=1000, viz = false)
# injected_heun, t_heun = main(Vx, Vy; nx=nx, ny=ny, α=1.0, nt=1000)

#  lines((cumsum(injected_rk2)); color=:red)
# lines!((cumsum(injected_heun)); color=:green)
# lines!((cumsum(injected_23)); color=:blue)

#  lines(cumsum(t_rk2); color=:red)
# lines!(cumsum(t_heun); color=:green)
# lines!(cumsum(t_23); color=:blue)

# df = DataFrame(
#     injected_23 = injected_23,
#     injected_rk2 = injected_rk2,
#     injected_heun = injected_heun,
#     t_23 = t_23,
#     t_rk2 = t_rk2,
#     t_heun = t_heun,
# )

# CSV.write("CPU_baseline.csv", df)

@generated function foo!(A::AbstractArray{T, N}, B::AbstractArray{T, N}) where {T,N}
    if N==2
        quote
           for j in axes(A,2), i in axes(A,1)
            A[i,j]=B[i,j]
           end
        end
    elseif N==3
        quote
           for k in axes(A,3), j in axes(A,2), i in axes(A,1)
            A[i,j,k]=B[i,j,k]
           end
        end
    end
end

@generated function foo2!(A::AbstractArray{T, N}, B::AbstractArray{T, N}) where {T,N}
    quote
        if $N==2
            for j in axes(A,2), i in axes(A,1)
                A[i,j]=B[i,j]
            end
        end
        if $N==3
            for k in axes(A,3), j in axes(A,2), i in axes(A,1)
                A[i,j,k]=B[i,j,k]
            end
        end
    end
end
 
function bar!(A::AbstractArray{T, N}, B::AbstractArray{T, N}) where {T,N}
    # if N==2
    #     for j in axes(A,2), i in axes(A,1)
    #         A[i,j]=B[i,j]
    #     end
    # elseif N==3
        for k in axes(A,3), j in axes(A,2), i in axes(A,1)
            A[i,j,k]=B[i,j,k]
        end
    # end
end

@code_warntype foo!(a,b)
@code_warntype foo2!(a,b)
@btime foo!($a,$b)
@btime foo2!($a,$b)
@btime bar!($a,$b)

n=128
a= rand(n,n)
b= rand(n,n)

a= rand(n,n,n)
b= rand(n,n,n)
