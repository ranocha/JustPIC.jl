
function shuffle_particles!(
    particles::Particles, grid, dxi, nxi::NTuple{N,T}, args
) where {N,T}
    # unpack
    (; coords, index, inject, max_xcell, min_xcell) = particles
    nx, ny = nxi
    px, py = coords

    offsets = ((1, 0, 0), (2, 0, 0), (1, 0, 1), (1, 1, 0))
    n_i = ceil(Int, nx * (1 / N))
    n_j = ceil(Int, ny * (1 / N))

    for offset_i in offsets
        offset, offset_x, offset_y = offset_i
        @parallel (1:n_i, 1:n_j) shuffle_particles_ps!(
            px,
            py,
            grid,
            dxi,
            nxi,
            index,
            inject,
            max_xcell,
            min_xcell,
            offset,
            offset_x,
            offset_y,
            args,
        )
    end

    # @assert (px, py) === particles.coords

    # (px, py) != particles.coords &&
    #     (@parallel (1:length(px)) copy_vectors!(particles.coords, (px, py)))

    return nothing
end

@parallel_indices (icell, jcell) function shuffle_particles_ps!(
    px,
    py,
    grid,
    dxi::NTuple{2,T},
    nxi,
    index,
    inject,
    max_xcell,
    min_xcell,
    offset,
    offset_x,
    offset_y,
    args,
) where {T}
    nx, ny = nxi
    i = offset + 2 * (icell - 1) + offset_x
    j = offset + 2 * (jcell - 1) + offset_y

    if (i ≤ nx) && (j ≤ ny)
        _shuffle_particles!(
            px, py, grid, dxi, nxi, index, inject, max_xcell, min_xcell, i, j, args
        )
    end
    return nothing
end

function _shuffle_particles!(
    px,
    py,
    grid,
    dxi,
    nxi,
    index,
    inject,
    max_xcell,
    min_xcell,
    icell,
    jcell,
    args::NTuple{N,T},
) where {N,T}

    nx, ny = nxi
    
    # closures --------------------------------------
    child_index(i,j) = (clamp(icell + i, 1, nx), clamp(icell + j, 1, ny))
    function find_free_memory(icell, jcell)
        for i in axes(index,1)
            index[i, icell, jcell] == 0 && return i
        end
        return 0
    end
    # -----------------------------------------------

    # coordinate of the lower-most-left coordinate of the parent cell 
    corner_xi = corner_coordinate(grid, icell, jcell)

    # # iterate over neighbouring (child) cells
    for i in -1:1, j in -1:1
        child = child_index(i,j) 

        # ignore parent cell
        if parent != child

            # iterate over particles in child cell 
            for ip in axes(px, 1)
                if index[ip, icell, jcell]
                    p_child = (px[ip, icell, jcell], py[ip, icell, jcell])

                    # check that the particle is inside the grid
                    if isincell(p_child, corner_xi, dxi)
                        # hold particle variables to move
                        current_px = px[ip, icell, jcell]
                        current_py = py[ip, icell, jcell]

                        (isnan(current_px) || isnan(current_py)) && continue

                        # cache out fields to move in the memory
                        current_args = ntuple(i -> args[i][ip, icell, jcell], Val(N))

                        # remove particle from old cell
                        index[ip, icell, jcell] = false
                        px[ip, icell, jcell] = NaN
                        py[ip, icell, jcell] = NaN

                        for t in eachindex(args)
                            args[t][ip, icell, jcell] = NaN
                        end

                        # move particle to new cell
                        free_idx = find_free_memory(icell, jcell)
                        # check whether current index is free
                        free_idx == 0 && continue 

                        # move it to the first free memory location
                        index[free_idx, icell, jcell] = true
                        px[free_idx, icell, jcell] = current_px
                        py[free_idx, icell, jcell] = current_py
                        # move fields in the memory
                        for t in eachindex(args)
                            args[t][free_idx, icell, jcell] = current_args[t]
                        end

                    end
                end
            end
        end
    end

    # true if cell is totally empty (i.e. we need to inject new particles in it)
    inject[icell, jcell] = isemptycell(icell, jcell, index, min_xcell)

    return nothing
end
