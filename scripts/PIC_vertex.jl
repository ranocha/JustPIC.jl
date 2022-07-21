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

# using StencilInterpolations
include(
    "/home/albert/Desktop/StencilInterpolations.jl/src/StencilInterpolations.jl"
)

using .StencilInterpolations

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
    Vx = Vx[1:end-1,:]
    Vy = Vy[:,1:end-1]
    Vx = PS_PACKAGE === :CUDA ? CuArray(Vx) : Vx
    Vy = PS_PACKAGE === :CUDA ? CuArray(Vy) : Vy

    V = (Vx, Vy)
    vx0, vy0 = maximum(abs.(Vx)), maximum(abs.(Vy))

    # model domain
    lx = ly = 10.0
    dx, dy = lx / (nx-1), ly / (ny-1)
    dxi = (dx, dy)
    nxi = (nx, ny)
    x = LinRange(dx / 2, lx - dx / 2, nx)
    y = LinRange(dy / 2, ly - dy / 2, ny)
    grid = (x, y)
    # velocity grids
    xv = 0:dx:lx
    yv = 0:dy:ly
    yvx = -dx/2:dy:ly+dy/2
    xvy = -dx/2:dx:lx+dx/2
    grid_vx = (xv, yvx)
    grid_vy = (xvy, yv)

    T = PS_PACKAGE === :CUDA ? CUDA.zeros(Float64, nx + 2, ny + 2) : zeros(nx, ny )
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
    gridv = (0:dx:lx, 0:dy:ly)
    gathering_xvertex!(T, pT, gridv, particles.coords)
    grid2particle_xvertex!(pT, gridv, T,  particles.coords)

    while it ≤ nt
        if it == nt ÷ 2
            dt = -dt
        end

        t1 = @elapsed begin
            copyto!(T0, T)

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

                #
                # we would run diffusion here
                #

                # grid to particles
                int2part!(argsi, T, T0, particles, grid)
                # grid2particle_xcell!(argsi, grid, T, particles.coords)
            end
        end

        it_time[it] = t1
        it += 1

        # viz && (it % nsave == 0) && plot(x, y, T, particles, pT, it)
    end

    return injected_cells, it_time
end

nx = ny = 42
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


# @inline @generated function foo(a::NTuple{N,T}, b::NTuple{N,T}, dxi::NTuple{N,T}) where {N,T}
#     quote
#         val = one(T)
#         Base.Cartesian.@nexprs $N i -> val *= one(T) - abs(a[i]-b[i])/dxi[i]
#         return val
#     end
# end

# @inbounds function _bar!(F, Fp, inode, jnode, xi, p, dxi)
#     px, py = p # particle coordinates
#     nx, ny = length.(xi)
#     xvertex = (xi[1][inode], xi[2][jnode]) # cell lower-left coordinates
#     ω, ωxF = 0.0, 0.0 # init weights
#     max_xcell = size(px, 1) # max particles per cell

#     # iterate over cells around i-th node
#     for ioffset in -1:0 
#         ivertex = ioffset + inode
#         for joffset in -1:0
#             jvertex = joffset +jnode
#             # make sure we stay within the grid
#             if (1 ≤ ivertex ≤ nx) && (1 ≤ jvertex ≤ ny)
#                 # iterate over cell
#                 for i in 1:max_xcell
#                     p_i = (px[i, inode, jnode], py[i, inode, jnode])
#                     # ignore lines below for unused allocations
#                     isnan(p_i[1]) && continue
#                     ω_i  = foo(xvertex, p_i, dxi)
#                     ω   += ω_i
#                     ωxF += ω_i*Fp[i, inode, jnode]
#                 end
#             end
#         end
#     end

#     F[inode, jnode] = ωxF/ω
# end

# function bar!(
#     F::Array{T,2}, Fp::AbstractArray{T}, xi, particle_coords
# ) where {T}
#     dxi = (
#         xi[1][2]-xi[1][1],
#         xi[2][2]-xi[2][1],
#     )
#     nx, ny = size(F)
#     Threads.@threads for jnode in 1:ny-1
#         for inode in 1:nx-1
#             _bar!(F, Fp, inode, jnode, xi, particle_coords, dxi)
#         end
#     end

# end

# gridv = (0:dx:lx, 0:dy:ly)
# bar!(T, pT, gridv, particles.coords)


function foo!(Fp, xvi, F::Array{T,N}, particle_coords) where {T,N}
    # cell dimensions
    dxi = StencilInterpolations.grid_size(xvi)
   
    nx, ny = length.(xvi)
    max_xcell = size(particle_coords[1], 1)
    Threads.@threads for jnode in 1:ny-1
        for inode in 1:nx-1
            _foo!(
                Fp, particle_coords, xvi, dxi, F, max_xcell, inode, jnode
            )
        end
    end
end


function _foo!(Fp, p::NTuple, xvi::NTuple, dxi::NTuple, F::AbstractArray, max_xcell, inode, jnode)
    idx = (inode, jnode)

    @inline function particle2tuple(ip::Integer, idx::NTuple{N,T}) where {N, T}
        return ntuple(i -> p[i][ip, idx...], Val(N))
    end

    for i in 1:max_xcell
        # check that the particle is inside the grid
        # isinside(p, xi)

        p_i = particle2tuple(i, idx)

        any(isnan, p_i) && continue

        # F at the cell corners
        Fi = StencilInterpolations.field_corners(F, idx)

        # normalize particle coordinates
        ti = StencilInterpolations.normalize_coordinates(p_i, xvi, dxi, idx)

        # Interpolate field F onto particle
        Fp[i, inode, jnode] = ndlinear(ti, Fi)

    end
end

foo!(pT, gridv, T,  particles.coords)
