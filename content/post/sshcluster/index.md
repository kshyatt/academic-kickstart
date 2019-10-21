---
title: "Running Cluster Jobs Remotely "
date: 2019-01-31T11:27:44-05:00
draft: false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors:
- admin

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags: []
categories: []

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["deep-learning"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
# projects = ["internal-project"]

---

## A Journey Through Laziness

Quite often, I'm working at my desktop in my campus office. The compute cluster on campus does not allow me to SSH directly to nodes or to spawn jobs, so usually you SSH into the head node and queue the jobs (which may be julia jobs, if I want to run some fancy parallel Julia code). 
 
This is kind of a pain, so I tried out using [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl) and julia's native `addprocs` and remote worker functionality to automate this a bit.

I have key-based login set up from `desktop` to `cluster` (the head node), and from `cluster` to its worker nodes.
Let's call my script `my_bad_science.jl` (I read somewhere that just like in the code itself, your writing about the code should contain
descriptive names).

First, I wanted to open a REPL on the desktop (I could make this a script too, of course). All the below examples are julia 0.5.0.
Then I opened an SSH tunnel to the cluster:

```julia
julia> addprocs([("cluster_hostname",1)], tunnel=true, max_parallel=1, exename="/home/kshyatt/julia/v0.5/julia", sshflags="-vv")
```

I passed `-vv` (verbose) to SSH because otherwise the Julia worker tends to time out. `exename` is there to tell julia that it's not installed in the same place on the cluster as it is on my desktop.

The output looks like:

```julia
# a bunch of verbose SSH chat happens here
# removed to protect the guilty/innocent
1-element Array{Int64,1}:
 2
```

Just like any other call to `addprocs`. Perfect! Now, on the remote head node, which is my worker, I will pull in my script:

```julia
# my bad science script

using ClusterManagers.jl

# our cluster uses PBS so
ClusterManagers.addprocs_pbs(12)

# now that the workers will have been loaded
# I need to import the packages I need to do
# the cool stuff

using Distributions
using research_utils # a private package of mine

#fill up some parameter arrays
Ls = collect(8:2:20)
w  = 10. # disorder strength
d  = Uniform(-w, w)
disorder = [rand(d, L) for L in Ls]

Hs = pmap(makeHamiltonians, Ls, disorder)   

# some code down here to write out the
# Hamiltonians to HDF5 - it's gross :(
```

What this will do is spawn a 12 worker job on the cluster, do the work, and write the results *on the cluster*. I could also have this script return something to the master worker on my desktop, if I wanted (perhaps timing information?).

Finally, on my desktop REPL, all I need to do is:

```julia
julia> rr = @spawnat 2 include("my_bad_science.jl")
Future(2,1,5,Nullable{Any}())

julia> wait(rr)
```

`@spawnat` returns a `Future`, and we need to `wait` for it to be finished. Or, we could `schedule` it and merrily go on our way, periodically checking to see if `rr` is finished.

One could also imagine a fancier version of this, plotting data as it comes in from the cluster head node on one's desktop (so we could use something a bit prettier than `UnicodePlots.jl`). For that you might need a `RemoteChannel`.

This is pretty short but it dramatically improved my lazing about, watching-job-results-come-in workflow. I hope it's useful to someone else!
