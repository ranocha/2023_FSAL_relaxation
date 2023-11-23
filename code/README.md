# Numerical experiments

This directory contains all code required to reproduce the numerical
experiments. First, you need to install Julia, e.g., by downloading
the binaries from the [download page](https://julialang.org/downloads/).
The numerical experiments were performed using Julia v1.9.3.

To reproduce the numerical experiments, start Julia and execute the
following commands in the Julia REPL.

```julia
julia> include("code.jl")

julia> plots_fsalr_section()

julia> plots_rfsal_section()

julia> plots_main()
```

Each function will create a separate directory where the plots will be saved.
This directory will be shown in the REPL.
