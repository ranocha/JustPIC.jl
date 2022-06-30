function lin2cart(cell, nx)
    ix = (cell-1) ÷ nx + 1
    iy = (cell-1) ÷ nx*(ix) + 1
    return ix, iy
end

function inject_particles(particles::Particles, grid, nxi, dxi)
    # closures 
    cell_number(i) = (i-1) ÷ (max_xcell) + 1
    first_cell_index(i) = (i - 1) * max_xcell + 1
    
    # unpack
    (; inject, coords, index, np, max_xcell) = particles
    nx, = nxi
    dx, dy = dxi

    for (i, injection) in enumerate(inject)
        if injection
            cell = cell_number(i)
            icell, jcell = lin2cart(cell, nx)
            xc, yc = corner_coordinate(grid, icell, jcell)

            # add 4 new particles in a 2x2 manner
            coords[1][index[i]] = xc + dx*(1/3)
            coords[2][index[i]] = yc
            coords[1][index[i+1]] = xc + dx*(2/3)
            coords[2][index[i+1]] = yc
            coords[1][index[i+2]] = xc + dx*(1/3)
            coords[2][index[i+2]] = yc + dy*(1/3)
            coords[1][index[i+3]] = xc + dx*(2/3)
            coords[2][index[i+3]] = yc + dy*(1/3)

            
            idx = first_cell_index(cell)

        end
    end
end



