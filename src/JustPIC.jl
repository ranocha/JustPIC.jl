module JustPIC

using MuladdMacro
using CUDA
using ParallelStencil
using StencilInterpolations

import StencilInterpolations: _grid2particle, parent_cell, isinside
export grid2particle!, gathering!

const PS_PACKAGE = Symbol(ENV["PS_PACKAGE"])

!ParallelStencil.is_initialized() && eval(:(@static if PS_PACKAGE == :CUDA
    @init_parallel_stencil(package = CUDA, ndims = 2)
    CUDA.allowscalar(true)
elseif PS_PACKAGE == :Threads
    @init_parallel_stencil(package = Threads, ndims = 2)
end))

include("particles.jl")
export Particles, init_particles, particle2grid! 

include("utils.jl")

include("advection.jl")
export advection_RK2!

include("injection.jl")
export inject_particles!, check_injection

include("shuffle.jl")
export shuffle_particles!

end # module
