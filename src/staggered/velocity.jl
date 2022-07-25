import .StencilInterpolations: normalize_coordinates, ndlinear

# INTERPOLATION METHODS

function _grid2particle_xcell_edge(
    p_i::NTuple, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, icell, jcell
)

    # cell index
    idx = (icell, jcell)
    # F at the cell corners
    Fi, xci = edge_nodes(F, p_i, xi_vx, dxi, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)

    return Fp
end

# Get field F at the centers of a given cell
@inline @inbounds function edge_nodes(
    F::AbstractArray{T,2}, p_i, xi_vx, dxi, idx::NTuple{2,Integer}
) where {T}
    # unpack
    idx_x, idx_y = idx
    px = p_i[1]
    dx = dxi[1]
    x_vx, y_vx = xi_vx
    xc = x_vx[idx_x]
    xv = xc + 0.5 * dx
    # compute offsets and corrections
    offset_x = (px - xv) > 0 ? 0 : 1
    # cell indices
    idx_x += offset_x
    # coordinates of lower-left corner of the cell
    xcell = x_vx[idx_x]
    ycell = y_vx[idx_y]

    # F at the four centers
    Fi = (
        F[idx_x, idx_y], F[idx_x + 1, idx_y], F[idx_x, idx_y + 1], F[idx_x + 1, idx_y + 1]
    )

    return Fi, (xcell, ycell)
end

# ADVECTION METHODS 

function advection_RK2_edges!(
    particles::Particles, V, grid_vx::NTuple{2,T}, grid_vy::NTuple{2,T}, dt, α
) where {T}
    # unpack 
    (; coords, index, max_xcell) = particles
    px, = coords
    # compute some basic stuff
    dxi = compute_dx(grid_vx)
    grid_lims = (
        extrema(grid_vx[1]) .+ (dxi[1] * 0.5, -dxi[1] * 0.5),
        extrema(grid_vy[2]) .+ (dxi[2] * 0.5, -dxi[2] * 0.5),
    )
    clamped_limits = clamp_grid_lims(grid_lims, dxi)
    _, nx, ny = size(px)
    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = (grid_vx, (grid_vy[2], grid_vy[1]))
    V_transp = (V[1], V[2]')
    # launch parallel advection kernel
    @parallel (1:max_xcell, 1:nx, 1:ny) advection_RK2_edges!(
        coords, V_transp, index, grid_vi, clamped_limits, dxi, dt, α
    )

    return nothing
end

@parallel_indices (ipart, icell, jcell) function advection_RK2_edges!(
    p,
    V::NTuple{2,AbstractArray{T,N}},
    index::AbstractArray,
    grid,
    clamped_limits,
    dxi,
    dt,
    α,
) where {T,N}
    px, py = p
    if icell ≤ size(px, 2) && jcell ≤ size(px, 3) && index[ipart, icell, jcell]
        pᵢ = (px[ipart, icell, jcell], py[ipart, icell, jcell])
        if !any(isnan, pᵢ)
            px[ipart, icell, jcell], py[ipart, icell, jcell] = _advection_RK2_edges(
                pᵢ, V, grid, dxi, clamped_limits, dt, icell, jcell; α=α
            )
        end
    end

    return nothing
end

"""
    y ← y + h*( (1-1/2/α)*f(t,y) + (1/2/α) * f(t, y+α*h*f(t,y)) )
    α = 0.5 ==> midpoint
    α = 1 ==> Heun
    α = 2/3 ==> Ralston
"""
function _advection_RK2_edges(
    p0::NTuple{N,T},
    v0::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    dxi,
    clamped_limits,
    dt,
    icell,
    jcell;
    α=0.5,
) where {T,N}
    _α = inv(α)
    ValN = Val(N)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        if i == 1
            _grid2particle_xcell_edge(p0, grid_vi[i], dxi, v0[i], icell, jcell)
        else
            _grid2particle_xcell_edge(
                (p0[2], p0[1]), grid_vi[i], (dxi[2], dxi[1]), v0[i], jcell, icell
            )
        end
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        xtmp = p0[i] + vp0[i] * α * dt
        clamp(xtmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        if i == 1
            _grid2particle_xcell_edge(p1, grid_vi[i], dxi, v0[i], icell, jcell)
        else
            _grid2particle_xcell_edge(
                (p1[2], p1[1]), grid_vi[i], (dxi[2], dxi[1]), v0[i], jcell, icell
            )
        end
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
