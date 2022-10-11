module JustPIC

using MuladdMacro
using ParallelStencil
using StencilInterpolations

const PS_PACKAGE = ENV["PS_PACKAGE"]

eval(:(@static PS_PACKAGE === "CUDA" && using CUDA))

!ParallelStencil.is_initialized() && eval(:(@static if PS_PACKAGE === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
elseif PS_PACKAGE == :Threads
    @init_parallel_stencil(Threads, Float64, 2)
end))

export gathering!, gathering_xcell!, gathering_xvertex!
export grid2particle!, grid2particle_xvertex!, grid2particle_xcell!

include("particles.jl")
export Particles, init_particles, particle2grid!

include("utils.jl")

include("advection.jl")
export advection_RK2!

include("injection.jl")
export inject_particles!, check_injection

include("shuffle.jl")
export shuffle_particles!

include("shuffle_vertex.jl")
export shuffle_particles_vertex!

include("staggered/centered.jl")
export int2part!, int2part_vertex!

include("staggered/velocity.jl")
export advection_RK2_edges!

end # module
