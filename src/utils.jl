# Color grid so that all elements containing the i-th node have a different color.
# During threaded assembly each thread acts upon a single color. In this way, we 
# can multithread along each color avoiding synchronisations between threads. It 
# could be extended amongst different cpus, where each cpu takes different colors
function neighbouring_cells(I::NTuple{N,T}, nxi::NTuple{N,T}) where {N,T}
    return neighbouring_cells(I..., nxi...)
end

# function neighbouring_cells(i, j, nx, ny)
#     # nx -= 1
#     # ny -= 1
#     nxi = (nx, ny)
#     idx = (
#         cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny)), nxi),
#         cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny)), nxi),
#         cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny)), nxi),
#         cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny)), nxi),
#         cart2lin((clamp(i + 1, 1, nx), clamp(j, 1, ny)), nxi),
#         cart2lin((clamp(i - 1, 1, nx), clamp(j + 1, 1, ny)), nxi),
#         cart2lin((clamp(i, 1, nx), clamp(j + 1, 1, ny)), nxi),
#         cart2lin((clamp(i + 1, 1, nx), clamp(j + 1, 1, ny)), nxi),
#     )
#     return idx
# end

# function neighbouring_cells(i, j, k, nx, ny, nz)
#     # nx -= 1
#     # ny -= 1
#     # nz -= 1
#     nxi = (nx, ny, nz)
#     idx = (
#         cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
#         cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
#         cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
#         cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k - 1, 1, nz)), nxi),
#         cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
#         cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
#         cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
#         cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k, 1, nz)), nxi),
#         cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
#         cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
#         cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
#         cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k + 1, 1, nz)), nxi),
#     )
#     return idx
# end

function neighbouring_cells(i, j, nx, ny)
    nxi = (nx, ny)
    idx = (
        (clamp(i - 1, 1, nx), clamp(j - 1, 1, ny)),
        (clamp(i    , 1, nx), clamp(j - 1, 1, ny)),
        (clamp(i + 1, 1, nx), clamp(j - 1, 1, ny)),
        (clamp(i - 1, 1, nx), clamp(j    , 1, ny)),
        (clamp(i + 1, 1, nx), clamp(j    , 1, ny)),
        (clamp(i - 1, 1, nx), clamp(j + 1, 1, ny)),
        (clamp(i    , 1, nx), clamp(j + 1, 1, ny)),
        (clamp(i + 1, 1, nx), clamp(j + 1, 1, ny)),
    )
    return idx
end

function neighbouring_cells(i, j, k, nx, ny, nz)
    nxi = (nx, ny, nz)
    idx = (
        cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
        cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k - 1, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k - 1, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
        cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
        cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny), clamp(k + 1, 1, nz)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny), clamp(k + 1, 1, nz)), nxi),
    )
    return idx
end

function color_cells(nxi::NTuple{2,T}) where {T}
    color_list = ntuple(Val(2)) do i0
        [cart2lin((i, j), nxi) for j in i0:2:nxi[2] for i in i0:2:nxi[1]]
    end
    return color_list
end

function color_cells(nxi::NTuple{3,T}) where {T}
    color_list = ntuple(Val(3)) do i0
        [cart2lin((i, j), nxi) for j in i0:3:nxi[3] for j in i0:3:nxi[2] for i in i0:3:nxi[1]]
    end
    return color_list
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

function isemptycell(
    idx::Integer, index::AbstractArray{T,N}, max_xcell::Integer, min_xcell::Integer
) where {T,N}
    # closures
    idx_range(i) = i:(i + max_xcell - 1)

    val = 0
    for j in idx_range(idx)
        @inbounds index[j] && (val += 1)
    end
    return val > min_xcell ? false : true
end

function isemptycell(
    icell::Integer, jcell::Integer, index::AbstractArray{T,N}, min_xcell::Integer
) where {T,N}

    val = 0
    for i in axes(index, 1)
        @inbounds index[i, icell, jcell] && (val += 1)
    end
    return val > min_xcell ? false : true
end

@parallel_indices (i) function copy_vectors!(
    dest::NTuple{N,T}, src::NTuple{N,T}
) where {N,T<:AbstractArray}
    for n in 1:N
        if i ≤ length(dest[n])
            dest[n][i] = src[n][i]
        end
    end
    return nothing
end