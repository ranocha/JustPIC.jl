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

# function inject_particles!(particles::Particles, grid, nxi, dxi)
#     # unpack
#     (; inject, coords, index, nxcell, max_xcell) = particles
#     # linear to cartesian object
#     i2s = CartesianIndices(nxi .- 1)
#     ncells = length(inject)
#     @parallel (1:ncells) inject_particles!(
#         inject, coords, index, nxcell, max_xcell, grid, dxi, i2s
#     )
# end

# @parallel_indices (cell) function inject_particles!(
#     inject, coords, index, nxcell, max_xcell, grid, dxi, i2s
# )
#     if cell ≤ length(inject)
#         _inject_particles!(inject, coords, index, nxcell, max_xcell, grid, dxi, i2s, cell)
#     end
#     return nothing
# end

# function _inject_particles!(inject, coords, index, nxcell, max_xcell, grid, dxi, i2s, cell)
#     dx, dy = dxi
#     # px, py = coords

#     # closures -----------------------------------
#     first_cell_index(i) = (i - 1) * max_xcell + 1
#     myrand() = (1.0 + (rand() - 0.5) * 0.25)
#     # --------------------------------------------

#     if inject[cell]
#         icell, jcell = i2s[cell].I
#         xc, yc = corner_coordinate(grid, icell, jcell)
#         idx = first_cell_index(cell)
#         # add 4 new particles in a 2x2 manner + some small random perturbation
#         coords[1][idx] = xc + dx * (1 / 3) * myrand()
#         coords[1][idx + 1] = xc + dx * (2 / 3) * myrand()
#         coords[1][idx + 2] = xc + dx * (1 / 3) * myrand()
#         coords[1][idx + 3] = xc + dx * (2 / 3) * myrand()
#         coords[2][idx] = yc + dy * (1 / 3) * myrand()
#         coords[2][idx + 1] = yc + dy * (1 / 3) * myrand()
#         coords[2][idx + 2] = yc + dy * (2 / 3) * myrand()
#         coords[2][idx + 3] = yc + dy * (2 / 3) * myrand()
#         for i in idx:(idx + nxcell - 1)
#             index[i] = true
#         end
#         inject[cell] = false
#     end
# end


@parallel_indices (icell, jcell) function check_injection!(inject, index, min_xcell)
    if icell ≤  size(index, 2) && icell ≤  size(index, 3)
        inject[icell, jcell] = isemptycell(icell, jcell, index, min_xcell)
    end
    return nothing
end

function check_injection(particles::Particles)

    (; inject, index, min_xcell) = particles
    nx, ny = size(particles.index, 2), size(particles.index, 3)
    
    @parallel (1:nx, 1:ny) check_injection!(inject, index, min_xcell)

    return check_injection(particles.inject)
end

@inline check_injection(inject::AbstractArray) = sum(inject) > 0 ? true : false

function inject_particles!(particles::Particles, grid, dxi)
    # unpack
    (; inject, coords, index, nxcell) = particles
    # linear to cartesian object
    icell, jcell = size(inject)
    @parallel (1:icell, 1:jcell) inject_particles!(
        inject, coords, index, grid, dxi, nxcell
    )
end

@parallel_indices (icell, jcell) function inject_particles!(
    inject, coords, index, grid, dxi, nxcell
)
    if (icell ≤ size(inject, 1)) && (jcell ≤ size(inject, 2))
        _inject_particles!(inject, coords, index, grid, dxi, nxcell, icell, jcell)
    end
    return nothing
end

function _inject_particles!(inject, coords, index, grid, dxi, nxcell, icell, jcell)
    dx, dy = dxi
    # px, py = coords
    max_xcell = size(index, 1)
    # closures -----------------------------------
    first_cell_index(i) = (i - 1) * max_xcell + 1
    myrand() = rand(-1:2:1) * rand() * 0.75
    # --------------------------------------------

    if inject[icell, jcell]

        particles_num = sum(index[i, icell, jcell] for i in 1:max_xcell)
        xc, yc = corner_coordinate(grid, icell, jcell)

        for i in 1:max_xcell
            if index[i, icell, jcell] == 0
                particles_num += 1
                # add at cellcenter + small random perturbation
                coords[1][i, icell, jcell] = xc + dx * 0.25 * myrand()
                coords[2][i, icell, jcell] = yc + dy * 0.25 * myrand()
                index[i, icell, jcell] = true
            end

            particles_num == nxcell && break
        end
    end
    inject[icell, jcell] = false

end
