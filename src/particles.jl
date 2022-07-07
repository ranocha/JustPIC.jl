# TODO preallocate buffers for gathering kernel
struct Particles{N,M,I,T1,T2,T3,T4}
    coords::NTuple{N,T1}
    index::T2
    inject::T3
    upper_buffer::T4
    lower_buffer::T4
    nxcell::I
    max_xcell::I
    min_xcell::I
    np::I

    function Particles(
        coords::NTuple{N,T1}, index, inject, nxcell::I, max_xcell::I, min_xcell::I, np::I, nxi
    ) where {N,I,T1}
                
        # types
        T2 = typeof(index)
        T3 = typeof(inject)
        T = eltype(coords[1])
        # allocate buffers for gathering kernel
        upper_buffer = if PS_PACKAGE === :CUDA
            CUDA.zeros(T, nxi...)
        else
            [Array{T, N}(undef, nxi...) for _ in 1:Threads.nthreads()]
        end
        lower_buffer = similar.(upper_buffer)
        T4 = typeof(lower_buffer)

        return new{N, max_xcell, I, T1, T2, T3, T4}(
            coords, index, inject, upper_buffer, lower_buffer, nxcell, max_xcell, min_xcell, np
        )
    end
end



function init_particles(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    rad2 = 2.0
    ncells = (nx - 1) * (ny - 1)
    np = max_xcell * ncells
    px, py, pT = ntuple(_ -> fill(NaN, np), Val(3))

    # index = zeros(UInt32, np)
    inject = falses(ncells)
    index = falses(np)
    @inbounds for j in 1:(ny - 1), i in 1:(nx - 1)
        # lowermost-left corner of the cell
        x0, y0 = x[i], y[j]
        # cell index
        cell = i + (nx - 1) * (j - 1)
        # index of first particle in cell
        idx = (cell - 1) * max_xcell + 1
        # add 4 new particles in a 2x2 manner + some small random perturbation
        px[idx] = x0 + dx * (1 / 3) * (1.0 + (rand() - 0.5))
        px[idx + 1] = x0 + dx * (2 / 3) * (1.0 + (rand() - 0.5))
        px[idx + 2] = x0 + dx * (1 / 3) * (1.0 + (rand() - 0.5))
        px[idx + 3] = x0 + dx * (2 / 3) * (1.0 + (rand() - 0.5))
        py[idx] = y0 + dy * (1 / 3) * (1.0 + (rand() - 0.5))
        py[idx + 1] = y0 + dy * (1 / 3) * (1.0 + (rand() - 0.5))
        py[idx + 2] = y0 + dy * (2 / 3) * (1.0 + (rand() - 0.5))
        py[idx + 3] = y0 + dy * (2 / 3) * (1.0 + (rand() - 0.5))
        # fill index array
        for l in idx:(idx + nxcell - 1)
            # index[l] = l
            index[l] = true
        end
    end

    if PS_PACKAGE === :CUDA
        pxi = CuArray.((px, py))
        return Particles(
            pxi, CuArray(index), CuArray(inject), nxcell, max_xcell, min_xcell, np, (nx, ny)
        )

    else
        return Particles((px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny))
    end
end