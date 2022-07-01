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

const PS_PACKAGE = :Threads

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
@inline cart2lin(i, j, nx) = i + (j - 1) * (nx - 1)
@inline cart2lin(i, j, k, nx, ny) = cart2lin(i, j, nx) + (k - 1) * (nx - 1) * (ny - 1)

function corner_coordinate(grid::NTuple{N,T1}, I::Vararg{T2,N}) where {T1,T2,N}
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

# function isemptycell(
#     cell::Integer, index::AbstractArray{T,N}, max_xcell::Integer
# ) where {T,N}
#     # closures
#     idx_range(i) = i:(i + max_xcell - 1)
#     first_cell_index(i) = (i - 1) * max_xcell + 1

#     idx = first_cell_index(cell)
#     for j in idx_range(idx)
#         if index[j]
#             return false
#         end
#     end
#     return true
# end

function shuffle_particles!(particles, grid, dxi, nxi)
    nx, ny = nxi
    # iterate over cells
    # for i in 1:2
    #     for icell in i:2:(nx - 1), jcell in i:2:(ny - 1)
    #         _shuffle_particles!(particles, grid, dxi, nxi, icell, jcell)
    #     end
    # end

    for icell in 1:(nx - 1), jcell in 1:(ny - 1)
        _shuffle_particles!(particles, grid, dxi, nxi, icell, jcell)
    end
end

function _shuffle_particles!(particles, grid, dxi, nxi, icell, jcell) 
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
    parent = cart2lin(jcell, icell, nx)
    # coordinate of the lower-most-left coordinate of the parent cell 
    corner_xi = corner_coordinate(grid, jcell, icell)
    i0_parent = first_cell_index(parent)

    # iterate over neighbouring (child) cells
    for child in neighbouring_cells(jcell, icell, nx, ny)

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

                        # remove particle from old cell
                        # index[j] = zero_T
                        index[j] = false
                        px[j] = NaN
                        py[j] = NaN

                        # move particle to new cell
                        free_idx = find_free_memory(idx_range(i0_parent))
                        free_idx == 0 && continue 

                        # move it to the first free memory location
                        index[free_idx] = true
                        # index[free_idx] = current_idx
                        px[free_idx] = current_px
                        py[free_idx] = current_py

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

function twoxtwo_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny)
    ncells = (nx - 1) * (ny - 1)
    np = max_xcell * ncells
    px, py = ntuple(_ -> fill(NaN, np), Val(2))

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
        end
    end

    return Particles((px, py), index, inject, nxcell, max_xcell, np)
end

function main(Vx, Vy; nx=40, ny=40, nxcell=4)
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

    # random particles
    max_xcell = nxcell * 2
    # particles = random_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny)
    particles = twoxtwo_particles(nxcell, max_xcell, x, y, dx, dy, nx, ny)

    # particle_coords_cpu = (px, py)
    # particle_coords = CuArray.(particle_coords_cpu)

    # field to interpolate
    dt = min(dx, dy) / max(abs(vx0), abs(vy0)) * 0.25
    nt = 1_000
    it = 1
    t = 0.0
    nsave = 10
    while it ≤ 50

        # while !check_injection(particles.inject) 
            # advect particles in space
            advection_RK2!(particles, V, grid, dxi, dt, 0.5)
            
            # advect particles in memory
            shuffle_particles!(particles, grid, dxi, nxi)
            
            println("$(sum(particles.inject)) empty cells")
        # end
        
        # inject particles in empty cell
        # check_injection(particles.inject) && 
        # inject_particles!(particles, grid, nxi, dxi)

        it += 1
        t += dt
        # if it % nsave == 0
        #     save_timestep!("out/step_$it.mat", particles.coords, t)
        # end
    end
    
    println("$(sum(particles.inject)) empty cells")

end

nx=ny=40
nxcell=4
Vx, Vy = load_benchmark_data("data/data41_benchmark.mat")
# main(Vx, Vy) # 426.857 μs (29 allocations: 172.78 KiB) 
# @btime main($Vx, $Vy) # 426.857 μs (29 allocations: 172.78 KiB) 

# # Vx, Vy = load_benchmark_data("data/data41_benchmark.mat")
# # Vx, Vy = CuArray(Vx), CuArray(Vy)
# # @btime main($Vx_dev, $Vy_dev) # 161.419 μs (213 allocat

first_cell_index(i) = (i - 1) * max_xcell + 1
idx_range(i) = i:(i + max_xcell - 1)

X = [x for x in x, y in y]
Y = [y for x in x, y in y]

px, py = particles.coords
pidx = particles.index
ii = findall(x->x==true, pidx)
scatter(px[ii], py[ii])
scatter!(vec(X),vec(Y))

ii=findall(particles.inject)
cell = ii[1]
icell, jcell = i2s[cell].I
xc, yc = corner_coordinate(grid, icell, jcell)
scatter!([xc], [yc], color=:blue, markersize = 10)

idx = idx_range(first_cell_index(cell))
scatter!(px[idx], py[idx], color=:red)


