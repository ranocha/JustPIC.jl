# function inject_particles!(particles::Particles, grid, nxi, dxi)
#     # unpack
#     (; inject, coords, nxcell, max_xcell) = particles
#     dx, dy = dxi
#     px, py = coords

#     # closures 
#     first_cell_index(i) = (i - 1) * max_xcell + 1
#     myrand() = (1.0 + (rand()-0.5)*0.25)

#     # linear to cartesian object
#     i2s = CartesianIndices(nxi.-1)

#     for (cell, injection) in enumerate(inject)
#         if injection
#             icell, jcell = i2s[cell].I
#             xc, yc = corner_coordinate(grid, icell, jcell)
#             idx = first_cell_index(cell)

#             # add 4 new particles in a 2x2 manner + some small random perturbation
#             px[idx]   = xc + dx*(1/3)*myrand()
#             px[idx+1] = xc + dx*(2/3)*myrand()
#             px[idx+2] = xc + dx*(1/3)*myrand()
#             px[idx+3] = xc + dx*(2/3)*myrand()
#             py[idx]   = yc + dy*(1/3)*myrand()
#             py[idx+1] = yc + dy*(1/3)*myrand()
#             py[idx+2] = yc + dy*(2/3)*myrand()
#             py[idx+3] = yc + dy*(2/3)*myrand()

#             for i in idx:(idx+nxcell-1)
#                 particles.index[i] = true
#             end

#             inject[cell] = false
#         end
#     end

# end


function inject_particles!(particles::Particles, grid, nxi, dxi)
    # unpack
    (; inject, coords, index, nxcell, max_xcell) = particles
    # linear to cartesian object
    i2s = CartesianIndices(nxi.-1)
    ncells = length(inject)
    @parallel (1:ncells) inject_particles!( inject, coords, index, nxcell, max_xcell, grid, dxi, i2s)
end

@parallel_indices (cell) function inject_particles!(inject, coords, index, nxcell, max_xcell, grid, dxi, i2s)
    if cell ≤ length(inject)
        _inject_particles!( inject, coords, index, nxcell, max_xcell, grid, dxi, i2s, cell)
    end
    return nothing
end

function _inject_particles!(inject, coords, index, nxcell, max_xcell, grid, dxi, i2s, cell)
   
    dx, dy = dxi
    px, py = coords

    # closures -----------------------------------
    first_cell_index(i) = (i - 1) * max_xcell + 1
    myrand() = (1.0 + (rand()-0.5)*0.25)
    # --------------------------------------------

    if inject[cell]
        icell, jcell = i2s[cell].I
        xc, yc = corner_coordinate(grid, icell, jcell)
        idx = first_cell_index(cell)
        # add 4 new particles in a 2x2 manner + some small random perturbation
        px[idx]   = xc + dx*(1/3)*myrand()
        px[idx+1] = xc + dx*(2/3)*myrand()
        px[idx+2] = xc + dx*(1/3)*myrand()
        px[idx+3] = xc + dx*(2/3)*myrand()
        py[idx]   = yc + dy*(1/3)*myrand()
        py[idx+1] = yc + dy*(1/3)*myrand()
        py[idx+2] = yc + dy*(2/3)*myrand()
        py[idx+3] = yc + dy*(2/3)*myrand()
        for i in idx:(idx+nxcell-1)
            index[i] = true
        end
        inject[cell] = false
    end

end

@inline check_injection(inject::AbstractArray) = sum(inject) > 0 ? true : false