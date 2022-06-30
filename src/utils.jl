# Color grid so that all elements containing the i-th node have a different color.
# During threaded assembly each thread acts upon a single color. In this way, we 
# can multithread along each color avoiding synchronisations between threads. It 
# could be extended amongst different cpus, where each cpu takes different colors
function neighbouring_cells(I::NTuple{N,T}, nxi::NTuple{N,T}) where {N,T}
    return neighbouring_cells(I..., nxi...)
end

function neighbouring_cells(i, j, nx, ny)
    nx -= 1
    ny -= 1
    nxi = (nx, ny)
    idx = (
        cart2lin((clamp(i - 1, 1, nx), clamp(j - 1, 1, ny)), nxi),
        cart2lin((clamp(i, 1, nx), clamp(j - 1, 1, ny)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j - 1, 1, ny)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j, 1, ny)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j, 1, ny)), nxi),
        cart2lin((clamp(i - 1, 1, nx), clamp(j + 1, 1, ny)), nxi),
        cart2lin((clamp(i, 1, nx), clamp(j + 1, 1, ny)), nxi),
        cart2lin((clamp(i + 1, 1, nx), clamp(j + 1, 1, ny)), nxi),
    )
    return idx
end

function neighbouring_cells(i, j, k, nx, ny, nz)
    nx -= 1
    ny -= 1
    nz -= 1
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
