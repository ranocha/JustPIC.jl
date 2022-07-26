function shuffle_particles_vertex!(
    particles::Particles, grid::NTuple{N,T}, args
) where {N,T}
    # unpack
    (; coords, index) = particles
    nxi = length.(grid)
    nx, ny = nxi
    px, py = coords
    dxi = (grid[1][2] - grid[1][1], grid[2][2] - grid[2][1])

    offsets = ((1, 0, 0), (2, 0, 0), (1, 0, 1), (1, 1, 0))
    n_i = ceil(Int, nx * (1 / N))
    n_j = ceil(Int, ny * (1 / N))

    for offset_i in offsets
        offset, offset_x, offset_y = offset_i
        @parallel (1:n_i, 1:n_j) shuffle_particles_vertex_ps!(
            px, py, grid, dxi, nxi, index, offset, offset_x, offset_y, args
        )
    end

    return nothing
end

@parallel_indices (icell, jcell) function shuffle_particles_vertex_ps!(
    px, py, grid, dxi::NTuple{2,T}, nxi, index, offset, offset_x, offset_y, args
) where {T}
    nx, ny = nxi
    i = offset + 2 * (icell - 1) + offset_x
    j = offset + 2 * (jcell - 1) + offset_y

    if (i ≤ nx - 1) && (j ≤ ny - 1)
        _shuffle_particles_vertex!(px, py, grid, dxi, nxi, index, i, j, args)
    end
    return nothing
end

function _shuffle_particles_vertex!(
    px, py, grid, dxi, nxi, index, icell, jcell, args::NTuple{N,T}
) where {N,T}
    nx, ny = nxi

    # closures --------------------------------------
    @inline child_index(i, j) = (icell + i, jcell + j)
    @inline function cache_args(
        args::NTuple{N1,T}, ip, child::Vararg{Int64,N2}
    ) where {T,N1,N2}
        return ntuple(i -> args[i][ip, child...], Val(N1))
    end
    @inline function find_free_memory(icell, jcell)
        for i in axes(index, 1)
            @inbounds index[i, icell, jcell] == 0 && return i
        end
        return 0
    end
    # -----------------------------------------------

    # coordinate of the lower-most-left coordinate of the parent cell 
    corner_xi = corner_coordinate(grid, icell, jcell)
    # cell where we check for incoming particles
    parent = icell, jcell
    # iterate over neighbouring (child) cells
    for j in -1:1, i in -1:1
        ichild, jchild = child_index(i, j)
        # ignore parent cell
        @inbounds if parent != (ichild, jchild) && (1 ≤ ichild ≤ nx - 1) && (1 ≤ jchild ≤ ny - 1)

            # iterate over particles in child cell 
            for ip in axes(px, 1)
                if index[ip, ichild, jchild] # true if memory allocation is filled with a particle
                    p_child = (px[ip, ichild, jchild], py[ip, ichild, jchild])

                    # check whether the incoming particle is inside the cell and move it
                    if isincell(p_child, corner_xi, dxi)

                        # hold particle variables
                        current_px = px[ip, ichild, jchild]
                        current_py = py[ip, ichild, jchild]
                        # cache out fields
                        current_args = cache_args(args, ip, ichild, jchild)

                        (isnan(current_px) || isnan(current_py)) && continue

                        # remove particle from child cell
                        index[ip, ichild, jchild] = false
                        px[ip, ichild, jchild] = NaN
                        py[ip, ichild, jchild] = NaN

                        for t in eachindex(args)
                            args[t][ip, ichild, jchild] = NaN
                        end

                        # check whether there's empty space in parent cell
                        free_idx = find_free_memory(icell, jcell)
                        free_idx == 0 && continue

                        # move it to the first free memory location
                        index[free_idx, icell, jcell] = true
                        px[free_idx, icell, jcell] = current_px
                        py[free_idx, icell, jcell] = current_py

                        # println("($ip, $icell, $jcell) moved to ($free_idx, $ichild, $jchild)")
                        # move fields in the memory
                        for t in eachindex(args)
                            args[t][free_idx, icell, jcell] = current_args[t]
                        end
                    end
                end
            end
        end
    end

    return nothing
end
