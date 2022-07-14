"""
    y ← y + h*( (1-1/2/α)*f(t,y) + (1/2/α) * f(t, y+α*h*f(t,y)) )
    α = 0.5 ==> midpoint
    α = 1 ==> Heun
    α = 2/3 ==> Ralston
"""
function _advection_RK2(
    p0::NTuple{N,T},
    v0::NTuple{N,AbstractArray{T,N}},
    grid,
    dxi,
    clamped_limits,
    dt,
    icell,
    jcell;
    α=0.5,
) where {T,N}
    _α = α
    ValN = Val(N)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        _grid2particle_xcell_centered(p0, grid, dxi, v0[i], icell, jcell)
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        xtmp = p0[i] + vp0[i] * α * dt
        clamp(xtmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        _grid2particle_xcell_centered(p1, grid, dxi, v0[i], icell, jcell)
    end

    # final advection
    pf = ntuple(ValN) do i
        ptmp = if α == 0.5
            @muladd p0[i] + dt * vp1[i]
        else
            @muladd p0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
        clamp(ptmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    return pf
end

# function advection_RK2!(particles::Particles, V, grid::NTuple{3,T}, dt, α) where {T}
#     (; coords, index, max_xcell) = particles
#     dxi = compute_dx(grid)
#     px, py, pz = coords
#     xci = minimum.(grid)
#     _, nx, ny, nz = size(px)

#     @parallel (1:max_xcell, 1:nx, 1:ny, 1:nz) advection_RK2!(px, py, pz, V, index, grid, xci, dxi, dt, α)

#     return nothing
# end

function advection_RK2!(particles::Particles, V, grid::NTuple{2,T}, dt, α) where {T}
    (; coords, index, max_xcell) = particles
    dxi = compute_dx(grid)
    px, py = coords
    # xci = minimum.(grid)
    grid_lims = extrema.(grid)
    clamped_limits = clamp_grid_lims(grid_lims, dxi)
    _, nx, ny = size(px)

    @parallel (1:max_xcell, 1:nx, 1:ny) advection_RK2!(
        px, py, V, index, grid, clamped_limits, dxi, dt, α
    )

    return nothing
end

# @parallel_indices (ipart, i, j, k) function advection_RK2!(
#     px, py, pz, V::NTuple{3,T1}, index::AbstractArray, grid, xci, dxi, dt, α
# ) where T1
#     if i ≤ size(px,2) && j ≤ size(px, 3) && k ≤ size(px, 4) && index[ipart, i, j, k]
#         pᵢ = (px[ipart, i, j, k], py[ipart, i, j, k], pz[ipart, i, j, k])
#         px[ipart, i, j, k], py[ipart, i, j, k], pz[ipart, i, j, k] = _advection_RK2(pᵢ, V, grid, xci, dxi, dt; α=α)
#     end

#     return nothing
# end

@parallel_indices (ipart, i, j) function advection_RK2!(
    px, py, V::NTuple{2,T1}, index::AbstractArray, grid, clamped_limits, dxi, dt, α
) where {T1}
    if i ≤ size(px, 2) && j ≤ size(px, 3) && index[ipart, i, j]
        pᵢ = (px[ipart, i, j], py[ipart, i, j])
        any(isnan, pᵢ) && continue
        px[ipart, i, j], py[ipart, i, j] = _advection_RK2(
            pᵢ, V, grid, dxi, clamped_limits, dt, i, j; α=α
        )
    end

    return nothing
end
