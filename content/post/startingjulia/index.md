---
title: "Getting Started With Julia"
date: 2019-02-01T10:45:08-05:00
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

Getting into a new programming language is always a bit overwhelming, because there are so many places to start! Here is a short list of things I like to send to people new to Julia, aimed at varying skill levels:

- [Learn X in Y Minutes - Julia edition](https://learnxinyminutes.com/docs/julia/), a good quick tour if you already know other languages like python pretty well
- [Towards Julia 1.0](https://www.youtube.com/watch?v=puMIBYthsjQ), by [David Sanders](https://github.com/dpsanders), is a great tutorial for the latest versions of Julia. Because the language went through so many changes before 1.0, not all older tutorials will still be usable!
- [Getting Started with 1.0's Package Manager](https://julialang.org/blog/2018/09/Pkgtutorial), for those you (probably everyone) who will want to start using libraries in the "package ecosystem" and maybe write your own!
- [How to use the new iteration interface](https://julialang.org/blog/2018/07/iterators-in-julia-0.7), since many people need to write `for`-loops!
- [Celeste.jl](https://github.com/jeff-regier/Celeste.jl) -- amazing example of what mixed (shared and distributed memory) parallelism is capable of in Julia
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), high performance small arrays which show off some of the features of julia's type parameterization
- [JuliaGPU](https://github.com/juliagpu), which has a collection of packages for writing native Julia code for GPUs and using existing GPU libraries from Julia
- [The JuliaLang learning page](https://julialang.org/learning/), which has lots of good links for learning the language and some best practices
