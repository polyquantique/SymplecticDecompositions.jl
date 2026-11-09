using Pkg

Pkg.activate(@__DIR__)

Pkg.add([
    "GenericLinearAlgebra",
    "GenericSchur",
    "MultiFloats",
    "RandomMatrices",
    "StatsBase",
])
