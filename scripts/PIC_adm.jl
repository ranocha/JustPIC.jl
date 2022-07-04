using MuladdMacro, MAT, CUDA
using ParallelStencil
using StencilInterpolations
import StencilInterpolations: _grid2particle, parent_cell, isinside

struct Particles{N, M, I, T1, T2, T3}
    coords::NTuple{N,T1}
    index::T2
    inject::T3
    nxcell::I
    max_xcell::I
    np::I

    function Particles(coords::NTuple{N,T1}, index, inject, nxcell, max_xcell, np) where {N,T1}
        I = typeof(np)
        T2 = typeof(index)
        T3 = typeof(inject)
        new{N, max_xcell, I, T1, T2, T3}(coords, index, inject, nxcell, max_xcell, np)
    end
end

push!(LOAD_PATH, "..")

const PS_PACKAGE = :CUDA

@static if PS_PACKAGE == :CUDA
    @init_parallel_stencil(package = CUDA, ndims = 2)
    CUDA.allowscalar(false)
elseif PS_PACKAGE == :Threads
    @init_parallel_stencil(package = Threads, ndims = 2)
end

include("../src/advection.jl")
include("../src/injection.jl")
include("../src/utils.jl")

## -------------------------------------------------------------

function load_benchmark_data(filename)
    params = matread(filename)
    return params["Vxp"], params["Vyp"]
end

function save_timestep!(fname, p, t)
    matwrite(
        fname, Dict("pX" => Array(p[1]), "pY" => Array(p[2]), "time" => t); compress=true
    )
    return nothing
end

@inline function cart2lin(I::NTuple{N,Integer}, nxi::NTuple{N,T}) where {N,T}
    return cart2lin(I..., ntuple(i -> nxi[i], Val(N - 1))...)
end
@inline cart2lin(i, j, nx) = i + (j - 1) * nx
@inline cart2lin(i, j, k, nx, ny) = cart2lin(i, j, nx) + (k - 1) * nx * ny

@inline function corner_coordinate(grid::NTuple{N,T1}, I::Vararg{T2,N}) where {T1,T2,N}
    return ntuple(i -> grid[i][I[i]], Val(N))
end

@inline function isincell(p::NTuple{2,T}, xci::NTuple{2,T}, dxi::NTuple{2,T}) where {T}
    px, py = p # particle coordinate
    xc, yc = xci # corner coordinate
    dx, dy = dxi # spacing between gridpoints

    # check if it's outside the x-limits
    px < xc && return false
    px > xc + dx && return false
    # check if it's outside the y-limits
    py < yc && return false
    py > yc + dy && return false
    # otherwise particle is inside parent cell
    return true
end

@inline function isincell(p::NTuple{3,T}, xci::NTuple{3,T}, dxi::NTuple{3,T}) where {T}
    px, py, pz = p # particle coordinate
    xc, yc, zc = xci # corner coordinate
    dx, dy, dz = dxi # spacing between gridpoints

    # check if it's outside the x- and y-limits
    !isincell((px, py), (xc, yc), (dx, dy)) && return false
    # check if it's outside the z-limits
    pz < zc && return false
    pz > zc + dz && return false
    # otherwise particle is inside the cell
    return true
end

function shuffle_particles!(particles, grid, dxi, nxi, args)
    nx, ny = nxi
    # iterate over cells
    # for i in 1:2
    #     for icell in i:2:(nx - 1), jcell in i:2:(ny - 1)
    #         _shuffle_particles!(particles, grid, dxi, nxi, icell, jcell)
    #     end
    # end

    for icell in 1:(nx - 1), jcell in 1:(ny - 1)
        _shuffle_particles!(particles, grid, dxi, nxi, icell, jcell, args)
    end
end

function _shuffle_particles!(particles, grid, dxi, nxi, icell, jcell, args::NTuple{N,T}) where {N,T}
    # unpack
    (; index, coords, inject, max_xcell) = particles
    px, py = coords
    nx, ny = nxi

    # closures --------------------------------------
    first_cell_index(i) = (i - 1) * max_xcell + 1
    idx_range(i) = i:(i + max_xcell - 1)
    
    function find_free_memory(indices)
        for i in indices
            index[i] == 0 && return i
        end
        return 0
    end
    # -----------------------------------------------

    # current (parent) cell (i.e. cell in the center of the cell-block)
    parent = cart2lin(icell, jcell, nx.-1)
    # coordinate of the lower-most-left coordinate of the parent cell 
    corner_xi = corner_coordinate(grid, icell, jcell)
    i0_parent = first_cell_index(parent)

    # iterate over neighbouring (child) cells
    for child in neighbouring_cells(icell, jcell, nx, ny)

        # ignore parent cell
        if parent != child
            # index where particles inside the child cell start in the particle array
            i0_child = first_cell_index(child)

            # iterate over particles in child cell 
            for j in idx_range(i0_child)

                if index[j]
                    p_child = (px[j], py[j])

                    # check that the particle is inside the grid
                    if isincell(p_child, corner_xi, dxi)
                        # hold particle variables to move
                        current_idx = j
                        # current_idx = index[j]
                        current_px = px[current_idx]
                        current_py = py[current_idx]

                        current_args = ntuple(Val(N)) do i
                            args[i][current_idx]
                        end

                        # remove particle from old cell
                        # index[j] = zero_T
                        index[j] = false
                        px[j] = NaN
                        py[j] = NaN

                        for k in eachindex(args)
                            args[k][j] = NaN
                        end

                        # move particle to new cell
                        free_idx = find_free_memory(idx_range(i0_parent))
                        free_idx == 0 && continue 

                        # move it to the first free memory location
                        index[free_idx] = true
                        # index[free_idx] = current_idx
                        px[free_idx] = current_px
                        py[free_idx] = current_py

                        for k in eachindex(args)
                            args[k][free_idx] = current_args[k]
                        end

                        # for k in idx_range(i0_parent)
                        #     # move it to the first free memory location
                        #     if index[k] == zero_T
                        #         index[k] = current_idx
                        #         px, py = coords

                        #         break
                        #     end
                        # end

                    end
                end
            end
        end
    end

    # true if cell is totally empty (i.e. we need to inject new particles in it)
    inject[parent] = isemptycell(i0_parent, index, max_xcell)

    return nothing
end

function isemptycell(
    idx::Integer, index::AbstractArray{T,N}, max_xcell::Integer
) where {T,N}
    # closures
    idx_range(i) = i:(i + max_xcell - 1)
    
    return sum(index[j] for  j in idx_range(idx)) > 0 ? false : true
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
        px[idx]   = x0 + dx*(1/3)*(1.0 + (rand()-0.5))
        px[idx+1] = x0 + dx*(2/3)*(1.0 + (rand()-0.5))
        px[idx+2] = x0 + dx*(1/3)*(1.0 + (rand()-0.5))
        px[idx+3] = x0 + dx*(2/3)*(1.0 + (rand()-0.5))
        py[idx]   = y0 + dy*(1/3)*(1.0 + (rand()-0.5))
        py[idx+1] = y0 + dy*(1/3)*(1.0 + (rand()-0.5))
        py[idx+2] = y0 + dy*(2/3)*(1.0 + (rand()-0.5))
        py[idx+3] = y0 + dy*(2/3)*(1.0 + (rand()-0.5))
        # fill index array
        for l in idx:(idx+nxcell-1) 
            # index[l] = l
            index[l] = true
            pT[l] = exp(
                -(
                    (x0 + px[l] * dx - lx / 2)^2 +
                    (y0 + py[l] * dy - ly / 2)^2
                ) / rad2,
            )
        end
    end

    if PS_PACKAGE === :CUDA
        pxi = CuArray.((px, py))
        return Particles(pxi, CuArray(index), CuArray(inject), nxcell, max_xcell, np), CuArray(pT)

    else
        return Particles((px, py), index, inject, nxcell, max_xcell, np), pT
    end
end

function foo(i)
    if i < 10
        return "0000$i"

    elseif 10 ≤ i ≤ 99
        return "000$i"

    elseif 100 ≤ i ≤ 999
        return "00$i"

    elseif 1000 ≤ i ≤ 9999
        return "0$i"
    end
end

function plot(x, y, T, particles, pT, it)

    pX, pY = particles.coords
    pidx = particles.index
    ii = findall(x->x==true, pidx)
    
    cmap = :jet

    f = Figure(resolution= (900, 450))
    ax1 = Axis(f[1,1])
    scatter!(ax1, pX[ii], pY[ii],color=pT[ii], colorrange=(0,1), colormap=cmap)
    
    ax2 = Axis(f[1,2])
    hm = heatmap!(ax2, x, y, T, colorrange=(0,1), colormap=cmap)
    Colorbar(f[1,3], hm)

    hideydecorations!(ax2)

    for ax in (ax1, ax2)
        xlims!(ax, 0, 10)
        ylims!(ax, 0, 10)
    end

    fi = foo(it)
    fname = joinpath("imgs", "fig_$(fi).png")
    save(fname, f)

    return f
end

function main(Vx, Vy; nx=40, ny=40, nxcell=4, α = 2/3, nt = 1_000)
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

    T = PS_PACKAGE === :CUDA ? CuMatrix{Float64}(undef, nx, ny) : Matrix{Float64}(undef, nx, ny)

    # random particles
    max_xcell = nxcell * 2
    # particles = random_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny)
    particles, pT = twoxtwo_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny, lx, ly)

    # particle_coords_cpu = (px, py)
    # particle_coords = CuArray.(particle_coords_cpu)
    pc = particles.coords

    # field to interpolate
    dt = min(dx, dy) / max(abs(vx0), abs(vy0)) * 0.25
    it = 1
    t = 0.0
    nsave = 10
    injected_cells = Vector{Int64}(undef,nt)
    it_time = Vector{Float64}(undef,nt)

    args = (pT, )
    while it ≤ nt

        t1 = @elapsed begin
            gathering!(T, pT, grid, pc)

            # advect particles in space
            advection_RK2!(particles, V, grid, dxi, dt, α)
            
            # advect particles in memory
            shuffle_particles!(particles, grid, dxi, nxi, args)
                
            # check_injection(particles.inject) && (
                # inject_particles!(particles, grid, nxi, dxi);
                # grid2particle!(pT, grid, T, particles.coords)
            # )

            check_injection(particles.inject) && inject_particles!(particles, grid, nxi, dxi)
            grid2particle!(pT, grid, T, particles.coords)

        end

        it_time[it] = t1

        # plot(x, y, T, particles, pT, it)

        it += 1
        t += dt
    end

    return injected_cells, it_time

end

nx=ny=40
nxcell=4
nt=1000
α = 2/3
Vx, Vy = load_benchmark_data("data/data41_benchmark.mat")
Vx, Vy = CuArray(Vx), CuArray(Vy)

injected_rk2, t_rk2  =  main(Vx, Vy; α = 0.5, nt=5000)
injected_heun, t_heun  =  main(Vx, Vy; α = 1.0, nt=5000)
injected_23, t_23 =  @time main(Vx, Vy; α = 2/3, nt=1000)

# @btime main($Vx, $Vy; α = $2/3, nt=$1000) #  1.202 s (156122 allocations: 109.18 MiB)

lines(cumsum(injected_rk2), color=:red)
lines!(cumsum(injected_heun), color=:green)
lines!(cumsum(injected_23), color=:blue)


lines(cumsum(t_rk2), color=:red)
lines!(cumsum(t_heun), color=:green)
lines!(cumsum(t_23), color=:blue)

# main(Vx, Vy) # 426.857 μs (29 allocations: 172.78 KiB) 
# @btime main($Vx, $Vy) # 426.857 μs (29 allocations: 172.78 KiB) 

# # Vx, Vy = load_benchmark_data("data/data41_benchmark.mat")
# # Vx, Vy = CuArray(Vx), CuArray(Vy)
# # @btime main($Vx_dev, $Vy_dev) # 161.419 μs (213 allocat

first_cell_index(i) = (i - 1) * max_xcell + 1
idx_range(i) = i:(i + max_xcell - 1)

parent = cart2lin(40, 1, nx)

X = [x for x in x, y in y]
Y = [y for x in x, y in y]

px, py = particles.coords
pidx = particles.index
ii = findall(x->x==true, pidx)
f,ax,s=scatter(px[ii], py[ii])
scatter!(ax,vec(X),vec(Y),color=:black)
f

ii=findall(particles.inject)
cell = ii[1]
icell, jcell = i2s[cell].I
xc, yc = corner_coordinate(grid, icell, jcell)
scatter!([xc], [yc], color=:yellow, markersize = 10)

idx = idx_range(first_cell_index(cell))
scatter!(px[idx], py[idx], color=:red)


