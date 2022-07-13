"""
    y ← y + h*( (1-1/2/α)*f(t,y) + (1/2/α) * f(t, y+α*h*f(t,y)) )
    α = 0.5 ==> midpoint
    α = 1 ==> Heun
    α = 2/3 ==> Ralston
"""
function _advection_RK2(
    x0::NTuple{N,T}, v0::NTuple{N,AbstractArray{T,N}}, grid, xci, dxi, dt; α=0.5
) where {T,N}
    _α = α
    grid_lims = extrema.(grid)
    ValN = Val(N)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        _grid2particle(x0, xci, grid, dxi, v0[i])
    end

    # advect α*dt
    x1 = ntuple(ValN) do i
        xtmp = x0[i] + vp0[i] * α * dt
        min_L, max_L = grid_lims[i]
        clamp(xtmp, min_L + dxi[i]*0.01, max_L - dxi[i]*0.01)
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        _grid2particle(x1, xci, grid, dxi, v0[i])
    end

    # final advection
    xf = ntuple(ValN) do i
        xtmp = if α == 0.5
            @muladd x0[i] + dt * vp1[i]
        else
            @muladd x0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
        min_L, max_L = grid_lims[i]
        clamp(xtmp, min_L + dxi[i]*0.01, max_L - dxi[i]*0.01)
    end

    return xf
end

function advection_RK2!(particles::Particles, V, grid::NTuple{3,T}, dt, α) where {T}
    (; coords, index, np) = particles
    dxi = ntuple(Val(3)) do i
        lo, hi = extrema(grid[i])
        hi - lo
    end
    px, py, pz = coords
    xci = minimum.(grid)

    @parallel (1:np) advection_RK2!(px, py, pz, V, index, grid, xci, dxi, dt, α)

    return nothing
end

function advection_RK2!(particles::Particles, V, grid::NTuple{2,T}, dt, α) where {T}
    (; coords, index, np) = particles
    dxi = ntuple(Val(2)) do i
        lo, hi = extrema(grid[i])
        hi - lo
    end
    px, py = coords
    xci = minimum.(grid)

    @parallel (1:np) advection_RK2!(px, py, V, index, grid, xci, dxi, dt, α)
    
    return nothing
end

@parallel_indices (i) function advection_RK2!(
    px, py, pz, V::NTuple{3,T1}, index::AbstractArray, grid, xci, dxi, dt, α
) where T1
    if i ≤ length(px) && index[i] === true
        pᵢ = (px[i], py[i], pz[i])
        px[i], py[i], pz[i] = _advection_RK2(pᵢ, V, grid, xci, dxi, dt; α=α)
    end

    return nothing
end

@parallel_indices (i) function advection_RK2!(
    px, py, V::NTuple{2,T1}, index::AbstractArray, grid, xci, dxi, dt, α
) where T1
    if i ≤ length(px) && index[i] === true
        pᵢ = (px[i], py[i])
        any(isnan, pᵢ) && continue
        px[i], py[i] = _advection_RK2(pᵢ, V, grid, xci, dxi, dt; α=α)
    end

    return nothing
end
