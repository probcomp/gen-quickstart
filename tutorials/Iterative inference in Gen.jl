# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Julia 1.1.1
#     language: julia
#     name: julia-1.1
# ---

# # Tutorial: Basics of Iterative Inference Programming in Gen

# This tutorial introduces the basics of inference programming in Gen using iterative inference programs, which include Markov chain Monte Carlo algorithms.

# ## The task: curve-fitting with outliers

# Suppose we have a dataset of points in the $x,y$ plane that is _mostly_ explained by a linear relationship, but which also has several outliers. Our goal will be to automatically identify the outliers, and to find a linear relationship (a slope and intercept, as well as an inherent noise level) that explains rest of the points:
#
# <img src="./images/example-inference.png" alt="See https://dspace.mit.edu/bitstream/handle/1721.1/119255/MIT-CSAIL-TR-2018-020.pdf, Figure 2(a))" width="600"/>
#
# This is a simple inference problem. But it has two features that make it ideal for introducing a couple new concepts in modeling and inference. First, we want not only to estimate the slope and intercept of the line that best fits the data, but also to classify each point as an inlier or outlier; that is, there are a large number of latent variables of interest, enough to make importance sampling an unreliable method (absent a more involved custom proposal that does the heavy lifting). Second, several of the parameters we're estimating (the slope and intercept) are continuous and amenable to gradient-based search techniques, which will allow us to explore Gen's optimization capabilities.
#
# Let's get started!

# ## Outline

# **Section 1.** [Writing the model: a first attempt](#writing-model)
#
# **Section 2.** [Visualizing the model's behavior](#visualizing)
#
# **Section 3.** [ The problem with generic importance sampling ](#importance)
#
# **Section 4.** [MCMC Inference Part 1: Block Resimulation](#mcmc-1)
#
# **Section 5.** [MCMC Inference Part 2: Gaussian Drift](#mcmc-2)
#
# **Section 6.** [MCMC Inference Part 3: Proposals based on heuristics](#mcmc-3)
#
# **Section 7.** [MAP Optimization](#map)

using Gen
import Random

# ## 1. Writing the model: a first attempt  <a name="writing-model"></a>

# We begin, as usual, by writing a model: a Julia function responsible (conceptually) for simulating a fake dataset.
#
# Our model will take as input a vector of `x` coordinates, and produce as output corresponding `y` coordinates. A simple approach to writing this model might look something like this:

@gen function model(xs::Vector{Float64})
    # First, generate some parameters of the model. We make these
    # random choices, because later, we will want to infer them
    # from data. The distributions we use here express our assumptions
    # about the parameters: we think the slope and intercept won't be
    # too far from 0; that the noise is relatively small; and that
    # the proportion of the dataset that don't fit a linear relationship
    # (outliers) could be anything between 0 and 1.
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    noise = @trace(gamma(1, 1), :noise)
    prob_outlier = @trace(uniform(0, 1), :prob_outlier)
    
    # Next, we generate the actual y coordinates.
    n = length(xs)
    ys = Vector{Float64}(undef, n)
    
    for i = 1:n
        # Decide whether this point is an outlier, and set
        # mean and standard deviation accordingly
        if @trace(bernoulli(prob_outlier), :data => i => :is_outlier)
            (mu, std) = (0., 10.)
        else
            (mu, std) = (xs[i] * slope + intercept, noise)
        end
        # Sample a y value for this point
        ys[i] = @trace(normal(mu, std), :data => i => :y)
    end
    ys
end;

# This model does what we want: it samples several parameters of the data-generating process, then generates data accordingly.

true_inlier_noise = 0.5
true_outlier_noise = 10.
prob_outlier = 0.1
true_slope = -1
true_intercept = 2
xs = collect(range(-5, stop=5, length=50))
ys = Float64[]
for (i, x) in enumerate(xs)
    if rand() < prob_outlier
        y = 0. + randn() * true_outlier_noise
    else
        y = true_slope * x + true_intercept + randn() * true_inlier_noise 
    end
    push!(ys, y)
end
ys[end-3] = 14
ys[end-5] = 13

# ## 2. What our model is doing: visualizing the prior 

# Let's visualize what our model is doing by drawing some samples from the prior. First, we'll need to write a function that serializes a trace for use by the GenViz library. Here, we make available the slope, intercept, noise level, and outlier classifications for each point, as these are the things that vary from trace to trace while doing inference.

# +
using GenViz

function serialize_trace(trace)
    assmt = Gen.get_choices(trace)
    (xs,) = Gen.get_args(trace)
    Dict("slope" => assmt[:slope],
        "intercept" => assmt[:intercept],
        "inlier_std" => assmt[:noise],
        "y-coords" => [assmt[:data => i => :y] for i in 1:length(xs)],
        "outliers" => [assmt[:data => i => :is_outlier] for i in 1:length(xs)])
end;
# -

# Next, we start a visualization server.

server = VizServer(8091);

# Finally, we generate some data and draw it:

# +
# Get some x coordinates and initialize a visualization
xs = collect(range(-5, stop=5, length=20))
viz = Viz(server, joinpath(@__DIR__, "regression-viz/dist"), [xs])

# Generate ten traces and draw them into the visualization
for i=1:10
    (trace, _) = Gen.generate(model, (xs,))
    ys = Gen.get_retval(trace)
    putTrace!(viz, "t$(i)", serialize_trace(trace))
end

# Display the visualization in this notebook
displayInNotebook(viz)
# -

# Note that an outlier can occur anywhere — including close to the line — and that our model is capable of generating datasets in which the vast majority of points are outliers.

# ## 3. The problem with generic importance sampling  <a name="generic-importance"></a>

# To motivate the need for more complex inference algorithms, let's begin by using the simple importance sampling method from the previous tutorial, and thinking about where it fails.
#
# First, let us create a synthetic dataset to do inference _about_.

# +
function make_synthetic_dataset(n)
    Random.seed!(1)
    prob_outlier = 0.2
    true_inlier_noise = 0.5
    true_outlier_noise = 5.0
    true_slope = -1
    true_intercept = 2
    xs = collect(range(-5, stop=5, length=n))
    ys = Float64[]
    for (i, x) in enumerate(xs)
        if rand() < prob_outlier
            y = randn() * true_outlier_noise
        else
            y = true_slope * x + true_intercept + randn() * true_inlier_noise
        end
        push!(ys, y)
    end
    (xs, ys)
end
    
(xs, ys) = make_synthetic_dataset(20);
# -

# In Gen, we express our _observations_ as an _Assignment_ that constrains the values of certain random choices to equal their observed values. Here, we want to constrain the values of the choices with address `:data => i => :y` (that is, the sampled $y$ coordinates) to equal the observed $y$ values. Let's write a helper function that takes in a vector of $y$ values and creates an Assignment that we can use to constrain our model:

function make_constraints(ys::Vector{Float64})
    constraints = Gen.choicemap()
    for i=1:length(ys)
        constraints[:data => i => :y] = ys[i]
    end
    constraints
end;

# We can apply it to our dataset's vector of `ys` to make a set of constraints for doing inference:

observations = make_constraints(ys);

# Now, we use the library function `importance_resampling` to draw approximate posterior samples given those observations:

function logmeanexp(scores)
    logsumexp(scores) - log(length(scores))
end

viz = Viz(server, joinpath(@__DIR__, "regression-viz/dist"), [xs])
log_probs = Vector{Float64}(undef, 10)
for i=1:10
    (tr, _) = Gen.importance_resampling(model, (xs,), observations, 2000)
    putTrace!(viz, "t$(i)", serialize_trace(tr))
    log_probs[i] = Gen.get_score(tr)
end
displayInNotebook(viz)
println("Average log probability: $(logmeanexp(log_probs))")

# We see here that importance sampling hasn't completely failed: it generally finds a reasonable position for the line. But the details are off: there is little logic to the outlier classification, and the inferred noise around the line is too wide. The problem is that there are just too many variables to get right, and so sampling everything in one go is highly unlikely to produce a perfect hit.
#
# In the remainder of this notebook, we'll explore techniques for finding the right solution _iteratively_, beginning with an initial guess and making many small changes, until we achieve a reasonable posterior sample.

# ## 4. MCMC Inference Part 1: Block Resimulation  <a name="mcmc-1"></a>

# ### What is MCMC?

# _Markov Chain Monte Carlo_ ("MCMC") methods are a powerful family of algorithms for iteratively producing approximate samples from a distribution (when applied to Bayesian inference problems, the posterior distribution of unknown (hidden) model variables given data).
#
# There is a rich theory behind MCMC methods, but we focus on applying MCMC in Gen and introducing theoretical ideas only when necessary for understanding. As we will see, Gen provides abstractions that hide and automate much of the math necessary for implementing MCMC algorithms correctly.
#
# The general shape of an MCMC algorithm is as follows. We begin by sampling an intial setting of all unobserved variables; in Gen, we produce an initial _trace_ consistent with (but not necessarily _probable_ given) our observations. Then, in a long-running loop, we make small, stochastic changes to the trace; in order for the algorithm to be asymptotically correct, these stochastic updates must satisfy certain probabilistic properties. 
#
# One common way of ensuring that the updates do satisfy those properties is to compute a _Metropolis-Hastings acceptance ratio_. Essentially, after proposing a change to a trace, we add an "accept or reject" step that stochastically decides whether to commit the update or to revert it. This is an over-simplification, but generally speaking, this step ensures we are more likely to accept changes that make our trace fit the observed data better, and to reject ones that make our current trace worse. The algorithm also tries not to go down dead ends: it is more likely to take an exploratory step into a low-probability region if it knows it can easily get back to where it came from.
#
# Gen's `metropolis_hastings` function _automatically_ adds this "accept/reject" check (including the correct computation of the probability of acceptance or rejection), so that as inference programmers, we need only think about what sorts of updates might be useful to propose. Starting in this section, we'll look at several design patterns for MCMC updates, and how to apply them in Gen.

# ### Block Resimulation

# One of the simplest strategies we can use is called Resimulation MH, and it works as follows.
#
# We begin, as in most iterative inference algorithms, by sampling an initial trace from our model, fixing the observed choices to their observed values.
#
# ```julia
# # Gen's `initialize` function accepts a model, a tuple of arguments to the model,
# # and an Assignment representing observations (or constraints to satisfy). It returns
# # a complete trace consistent with the observations, and an importance weight.
# (tr, _) = initialize(model, (xs,), observations)
# ```

# Then, in each iteration of our program, we propose changes to all our model's variables in "blocks," by erasing a set of variables from our current trace and _resimulating_ them from the model. After resimulating each block of choices, we perform an accept/reject step, deciding whether the proposed changes are worth making. 
#
# ```julia
# # Pseudocode
# for iter=1:500
#     tr = maybe_update_block_1(tr)
#     tr = maybe_update_block_2(tr)
#     ...
#     tr = maybe_update_block_n(tr)
# end
# ```
#
# The main design choice in designing a Block Resimulation MH algorithm is how to block the choices together for resimulation. At one extreme, we could put each random choice the model makes in its own block. At the other, we could put all variables into a single block (a strategy sometimes called "independent" MH, and which bears a strong similarity to importance resampling, as it involves repeatedly generating completely new traces and deciding whether to keep them or not). Usually, the right thing to do is somewhere in between.
#
# For the regression problem, here is one possible blocking of choices:
#
# **Block 1: `slope`, `intercept`, and `noise`.** These parameters determine the linear relationship; resimulating them is like picking a new line. We know from our importance sampling experiment above that before too long, we're bound to sample something close to the right line.
#
# **Blocks 2 through N+1: Each `is_outlier`, in its own block.** One problem we saw with importance sampling in this problem was that it tried to sample _every_ outlier classification at once, when in reality the chances of a single sample that correctly classifies all the points are very low. Here, we can choose to resimulate each `is_outlier` choice separately, and for each one, decide whether to use the resimulated value or not.
#
# **Block N+2: `prob_outlier`.** Finally, we can propose a new `prob_outlier` value; in general, we can expect to accept the proposal when it is line with the current hypothesized proportion of `is_outlier` choices that are set to `true`.
#
# Resimulating a block of variables is the simplest form of update that Gen's `metropolis_hastings` operator (or `mh` for short) supports. When supplied with a _current trace_ and a _selection_ of trace addresses to resimulate, `mh` performs the resimulation and the appropriate accept/reject check, then returns a possibly updated trace. A selection is created using the `select` method. So a single update of the scheme we proposed above would look like this:

# Perform a single block resimulation update of a trace.
function block_resimulation_update(tr)
    # Block 1: Update the line's parameters
    line_params = select(:noise, :slope, :intercept)
    (tr, _) = mh(tr, line_params)
    
    # Blocks 2-N+1: Update the outlier classifications
    (xs,) = get_args(tr)
    n = length(xs)
    for i=1:n
        (tr, _) = mh(tr, select(:data => i => :is_outlier))
    end
    
    # Block N+2: Update the prob_outlier parameter
    (tr, _) = mh(tr, select(:prob_outlier))
    
    # Return the updated trace
    tr
end;

# All that's left is to (a) obtain an initial trace, and then (b) run that update in a loop for as long as we'd like:

function block_resimulation_inference(xs, ys)
    observations = make_constraints(ys)
    (tr, _) = generate(model, (xs,), observations)
    for iter=1:500
        tr = block_resimulation_update(tr)
    end
    tr
end;

# Let's test it out:

scores = Vector{Float64}(undef, 10)
for i=1:10
    @time tr = block_resimulation_inference(xs, ys)
    scores[i] = get_score(tr)
end
println("Log probability: ", logmeanexp(scores))

# We note that this is significantly better than importance sampling, even if we run importance sampling for about the same amount of (wall-clock) time per sample:

scores = Vector{Float64}(undef, 10)
for i=1:10
    @time (tr, _) = importance_resampling(model, (xs,), observations, 17000)
    scores[i] = get_score(tr)
end
println("Log probability: ", logmeanexp(scores))

# It's one thing to see a log probability increase; it's better to understand what the inference algorithm is actually doing, and to see _why_ it's doing better.
#
# A great tool for debugging and improving MCMC algorithms is visualization. We can use GenViz's `displayInNotebook(viz) do ... end` syntax to produce an animated visualization:

viz = Viz(server, joinpath(@__DIR__, "regression-viz/dist"), [xs, ys])
Random.seed!(2)
displayInNotebook(viz) do
    (tr, _) = generate(model, (xs,), observations)
    putTrace!(viz, "t", serialize_trace(tr))
    for iter = 1:500
        tr = block_resimulation_update(tr)
        
        # Visualize and sleep for clearer animation
        putTrace!(viz, "t", serialize_trace(tr))
        sleep(0.01)
    end
end

# ## 5. MCMC Inference Part 2: Gaussian Drift MH  <a name="mcmc-2"></a>

# So far, we've seen one form of incremental trace update:
#
# ```julia
# (tr, did_accept) = mh(tr, select(:address1, :address2, ...))
# ```
#
# This update is incremental in that it only proposes changes to part of a trace (the selected addresses). But when computing _what_ changes to propose, it ignores the current state completely and resimulates all-new values from the model.
#
# That wholesale resimulation of values is often not the best way to search for improvements. To that end, Gen also offers a more general flavor of MH:
#
# ```julia
# (tr, did_accept) = mh(tr, custom_proposal, custom_proposal_args)
# ```
#
# A "custom proposal" is just what it sounds like: whereas before, we were using the _default resimulation proposal_ to come up with new values for the selected addresses, we can now pass in a generative function that samples proposed values however it wants.
#
# For example, here is a custom proposal that takes in a current trace, and proposes a new slope and intercept by randomly perturbing the existing values:

@gen function line_proposal(trace)
    choices = get_choices(trace)
    slope = @trace(normal(choices[:slope], 0.5), :slope)
    intercept = @trace(normal(choices[:intercept], 0.5), :intercept)
end;

# This is often called a "Gaussian drift" proposal, because it essentially amounts to proposing steps of a random walk. (What makes it different from a random walk is that we will still use an MH accept/reject step to make sure we don't wander into areas of very low probability.)

# To use the proposal, we write:
#
# ```julia
# (tr, did_accept) = mh(tr, line_proposal, ())
# ```
#
# Two things to note:
# 1. We no longer need to pass a selection of addresses. Instead, Gen assumes that whichever addresses are sampled by the proposal (in this case, `:slope` and `:intercept`) are being proposed to.
# 2. The argument list to the proposal is an empty tuple, `()`. The `line_proposal` generative function does expect an argument, the previous trace, but this is supplied automatically to all MH custom proposals.

# Let's swap it into our update:

function gaussian_drift_update(tr)
    # Gaussian drift on line params
    (tr, _) = mh(tr, line_proposal, ())
    
    # Block resimulation: Update the outlier classifications
    (xs,) = get_args(tr)
    n = length(xs)
    for i=1:n
        (tr, _) = mh(tr, select(:data => i => :is_outlier))
    end
    
    # Block resimulation: Update the prob_outlier parameter
    (tr, w) = mh(tr, select(:prob_outlier))
    (tr, w) = mh(tr, select(:noise))
    tr
end;

# If we compare the Gaussian Drift proposal visually with our old algorithm, we can see how the new behavior helps:

# +
viz = Viz(server, joinpath(@__DIR__, "regression-viz/dist"), [xs, ys])

# Set random seed for a reproducible animation
Random.seed!(35)

# Create the animation
displayInNotebook(viz) do
    # Get an initial trace
    (tr1, _) = generate(model, (xs,), observations)
    tr2 = tr1
    
    # Visualize the initial trace twice
    putTrace!(viz, 1, serialize_trace(tr1))
    putTrace!(viz, 2, serialize_trace(tr2))
    sleep(1)
    
    # Improve both traces
    for iter = 1:300
        # Gaussian drift update in trace 1
        tr1 = gaussian_drift_update(tr1)
        # Block resimulation update in trace 2
        tr2 = block_resimulation_update(tr2)
        
        # Visualize and sleep for clearer animation
        putTrace!(viz, 1, serialize_trace(tr1))
        putTrace!(viz, 2, serialize_trace(tr2))
        sleep(0.02)
    end
end
# -

# <hr />

# ### Exercise: Analyzing the algorithms

# Run the cell above several times with different random seeds. Compare the two algorithms with respect to the following:
#
# - How fast do they find a relatively good line?
#
# - Does one of them tend to get stuck more than the other? Under what conditions? Why?

# *Your answer here*

# <hr />

# A more quantitative comparison demonstrates that our change has definitely improved our inference quality:

# +
Random.seed!(1)
function gaussian_drift_inference()
    (tr, _) = generate(model, (xs,), observations)
    for iter=1:500
        tr = gaussian_drift_update(tr)
    end
    tr
end

scores = Vector{Float64}(undef, 10)
for i=1:10
    @time tr = gaussian_drift_inference()
    scores[i] = get_score(tr)
end
println("Log probability: ", logmeanexp(scores))
# -

# ## 6. MCMC Inference Part 3: Heuristics to guide the process  <a name="mcmc-3"></a>

# In this section, we'll look at another strategy for improving MCMC inference: using arbitrary heuristics to make smarter proposals. In particular, we'll use a method called "Random Sample Consensus" (or RANSAC) to quickly find promising settings of the slope and intercept parameters.
#
# RANSAC works as follows:
# 1. We repeatedly choose a small random subset of the points, say, of size 3.
# 2. We do least-squares linear regression to find a line of best fit for those points.
# 3. We count how many points (from the entire set) are near the line we found.
# 4. After a suitable number of iterations (say, 10), we return the line that had the highest score.
#
# Here's our implementation of the algorithm in Julia:

# +
import StatsBase

struct RANSACParams
    # the number of random subsets to try
    iters::Int

    # the number of points to use to construct a hypothesis
    subset_size::Int

    # the error threshold below which a datum is considered an inlier
    eps::Float64
    
    function RANSACParams(iters, subset_size, eps)
        if iters < 1
            error("iters < 1")
        end
        new(iters, subset_size, eps)
    end
end


function ransac(xs::Vector{Float64}, ys::Vector{Float64}, params::RANSACParams)
    best_num_inliers::Int = -1
    best_slope::Float64 = NaN
    best_intercept::Float64 = NaN
    for i=1:params.iters
        # select a random subset of points
        rand_ind = StatsBase.sample(1:length(xs), params.subset_size, replace=false)
        subset_xs = xs[rand_ind]
        subset_ys = ys[rand_ind]
        
        # estimate slope and intercept using least squares
        A = hcat(subset_xs, ones(length(subset_xs)))
        slope, intercept = A\subset_ys
        
        ypred = intercept .+ slope * xs

        # count the number of inliers for this (slope, intercept) hypothesis
        inliers = abs.(ys - ypred) .< params.eps
        num_inliers = sum(inliers)

        if num_inliers > best_num_inliers
            best_slope, best_intercept = slope, intercept
            best_num_inliers = num_inliers
        end
    end

    # return the hypothesis that resulted in the most inliers
    (best_slope, best_intercept)
end;
# -

# We can now wrap it in a Gen proposal that calls out to RANSAC, then samples a slope and intercept near the one it proposed.

@gen function ransac_proposal(prev_trace, xs, ys)
    (slope, intercept) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    @trace(normal(slope, 0.1), :slope)
    @trace(normal(intercept, 1.0), :intercept)
end;

# (Notice that although `ransac` makes random choices, they are not addressed (and they happen outside of a Gen generative function), so Gen cannot reason about them. This is OK (see [Using probabilistic programs as proposals](https://arxiv.org/abs/1801.03612)). Writing proposals that have traced internal randomness (i.e., that make traced random choices that are not directly used in the proposal) can lead to better inference, but requires the use of a more complex version of Gen's `mh` operator, which is beyond the scope of this tutorial.)

# One iteration of our update algorithm will now look like this:

function ransac_update(tr)
    # Use RANSAC to (potentially) jump to a better line
    # from wherever we are
    (tr, _) = mh(tr, ransac_proposal, (xs, ys))
    
    # Spend a while refining the parameters, using Gaussian drift
    # to tune the slope and intercept, and resimulation for the noise
    # and outliers.
    for j=1:20
        (tr, _) = mh(tr, select(:prob_outlier))
        (tr, _) = mh(tr, select(:noise))
        (tr, _) = mh(tr, line_proposal, ())
        # Reclassify outliers
        for i=1:length(get_args(tr)[1])
            (tr, _) = mh(tr, select(:data => i => :is_outlier))
        end
    end
    tr
end

# We can now run our main loop for just 5 iterations, and achieve pretty good results. (Of course, since we do 20 inner loop iterations in `ransac_update`, this is really closer to 100 iterations.) The running time is significantly lower than before, without a real dip in quality:

# +
function ransac_inference()
    (slope, intercept) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    slope_intercept_init = choicemap()
    slope_intercept_init[:slope] = slope
    slope_intercept_init[:intercept] = intercept
    (tr, _) = generate(model, (xs,), merge(observations, slope_intercept_init))
    for iter=1:5
        tr = ransac_update(tr)
    end
    tr
end

scores = Vector{Float64}(undef, 10)
for i=1:10
    @time tr = ransac_inference()
    scores[i] = get_score(tr)
end
println("Log probability: ", logmeanexp(scores))
# -

# Let's visualize the algorithm:

viz = Viz(server, joinpath(@__DIR__, "regression-viz/dist"), [xs, ys])
displayInNotebook(viz) do
    (slope, intercept) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    slope_intercept_init = choicemap()
    slope_intercept_init[:slope] = slope
    slope_intercept_init[:intercept] = intercept
    (tr, _) = generate(model, (xs,), merge(observations, slope_intercept_init))
    putTrace!(viz, "t", serialize_trace(tr))
    for iter = 1:5
        (tr, _) = mh(tr, ransac_proposal, (xs, ys))
    
        # Spend a while refining the parameters, using Gaussian drift
        # to tune the slope and intercept, and resimulation for the noise
        # and outliers.
        for j=1:20
            (tr, _) = mh(tr, select(:prob_outlier))
            (tr, _) = mh(tr, select(:noise))
            (tr, _) = mh(tr, line_proposal, ())
            # Reclassify outliers
            for i=1:length(get_args(tr)[1])
                (tr, _) = mh(tr, select(:data => i => :is_outlier))
            end
            putTrace!(viz, "t", serialize_trace(tr))
            sleep(0.1)
        end
    end
end

# <hr />
#
# ### Exercise: Improving the heuristic
# Currently, the RANSAC heuristic does not use the current trace's information at all. Try changing it to use the current state as follows:
#
# Instead of a constant `eps` parameter that controls whether a point is considered an inlier, make this decision based on the currently hypothesized `noise` level.
#
# Does this improve the inference? Do you have a guess as to why?
# <hr />
#
# ### Exercise: A different heuristic
# Implement a heuristic-based proposal that selects the points that are currently classified as _inliers_, finds the line of best fit, and adds some noise.
# <hr />

# ### Exercise: Initialization
#
# In our inference program above, when generating an initial trace on which to iterate, we initialize the slope and intercept to values proposed by RANSAC. If we don't do this, the performance decreases sharply, despite the fact that we still propose new slope/intercept pairs from RANSAC once the loop starts. Why is this?

# ## 7. MAP Optimization  <a name="map"></a>

# Everything we've done so far has been within the MCMC framework. But sometimes you're not interested in getting posterior samples—sometimes you just want a single likely explanation for your data. Gen also provides tools for _maximum a posteriori_ estimation ("MAP estimation"), the problem of finding a trace that maximizes the posterior probability under the model given observations.

# For example, let's say we wanted to take a trace and assign each point's `is_outlier` score to the most likely possibility. We can do this by iterating over both possible traces, scoring them, and choosing the one with the higher score. We can do this using Gen's [`update`](https://probcomp.github.io/Gen/dev/ref/gfi/#Update-1) function, which allows us to manually update a trace to satisfy some constraints:

function is_outlier_map_update(tr)
    (xs,) = get_args(tr)
    for i=1:length(xs)
        constraints = choicemap(:prob_outlier => 0.1)
        constraints[:data => i => :is_outlier] = false
        (trace1,) = update(tr, (xs,), (NoChange(),), constraints)
        constraints[:data => i => :is_outlier] = true
        (trace2,) = update(tr, (xs,), (NoChange(),), constraints)
        tr = (get_score(trace1) > get_score(trace2)) ? trace1 : trace2
    end
    tr
end

# For continuous parameters, we can use Gen's `map_optimize` function, which uses automatic differentiation to shift the selected parameters in the direction that causes the probability of the trace to increase most sharply:
#
# ```julia
# tr = map_optimize(tr, select(:slope, :intercept), max_step_size=1., min_step_size=1e-5)
# ```
#
# Putting these updates together, we can write an inference program that uses our RANSAC algorithm from above to get an initial trace, then tunes it using optimization:

# +
viz = Viz(server, joinpath(@__DIR__, "regression-viz/dist"), [xs, ys])
ransac_score, final_score = displayInNotebook(viz) do
    (slope, intercept) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    slope_intercept_init = choicemap()
    slope_intercept_init[:slope] = slope
    slope_intercept_init[:intercept] = intercept
    (tr,) = generate(model, (xs,), merge(observations, slope_intercept_init))
    sleep(1)
    putTrace!(viz, "t", serialize_trace(tr))
    for iter=1:5
        tr = ransac_update(tr)
        putTrace!(viz, "t", serialize_trace(tr))
        sleep(0.1)
    end
    ransac_score = get_score(tr)
    sleep(1)
    for iter = 1:30
        # Take a single gradient step on the line parameters.
        tr = map_optimize(tr, select(:slope, :intercept), max_step_size=1., min_step_size=1e-5)
        tr = map_optimize(tr, select(:noise), max_step_size=1., min_step_size=1e-5)
        
        # Choose the most likely classification of outliers.
        tr = is_outlier_map_update(tr)
        
        # Update the prob outlier
        choices = get_choices(tr)
        optimal_prob_outlier = count(i -> choices[:data => i => :is_outlier], 1:length(xs)) / length(xs)
        optimal_prob_outlier = min(0.5, max(0.05, optimal_prob_outlier))
        (tr, _) = update(tr, (xs,), (NoChange(),), choicemap(:prob_outlier => optimal_prob_outlier))
        
        # Visualize and sleep for clearer animation
        putTrace!(viz, "t", serialize_trace(tr))
        sleep(0.1)
    end
    final_score = get_score(tr)
    ransac_score, final_score
end

println("Score after ransac: $(ransac_score). Final score: $(final_score).")
# -

# Below, we evaluate the algorithm and we see that it gets our best scores yet, which is what it's meant to do:

# +
function map_inference()
    (slope, intercept) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    slope_intercept_init = choicemap()
    slope_intercept_init[:slope] = slope
    slope_intercept_init[:intercept] = intercept
    (tr, _) = generate(model, (xs,), merge(observations, slope_intercept_init))
    for iter=1:5
        tr = ransac_update(tr)
    end
    
    for iter = 1:20
        # Take a single gradient step on the line parameters.
        tr = map_optimize(tr, select(:slope, :intercept), max_step_size=1., min_step_size=1e-5)
        tr = map_optimize(tr, select(:noise), max_step_size=1., min_step_size=1e-5)
        
        # Choose the most likely classification of outliers.
        tr = is_outlier_map_update(tr)
        
        # Update the prob outlier
        choices = get_choices(tr)
        optimal_prob_outlier = count(i -> choices[:data => i => :is_outlier], 1:length(xs)) / length(xs)
        optimal_prob_outlier = min(0.5, max(0.05, optimal_prob_outlier))
        (tr, _) = update(tr, (xs,), (NoChange(),), choicemap(:prob_outlier => optimal_prob_outlier))        
    end
    tr
end

scores = Vector{Float64}(undef, 10)
for i=1:10
    @time tr = map_inference()
    scores[i] = get_score(tr)
end
println(logmeanexp(scores))
# -

# This doesn't necessarily mean that it's "better," though. It finds the most probable explanation of the data, which is a different problem from the one we tackled with MCMC inference. There, the goal was to sample from the posterior, which allows us to better characterize our uncertainty. Using MCMC, there might be a borderline point that is sometimes classified as an outlier and sometimes not, reflecting our uncertainty; with MAP optimization, we will always be shown the most probable answer.

# <hr />
#
# ## Exercise: Bimodal posterior
# Generate a dataset for which there are two distinct possible explanations under our model. Then, for each algorithm discussed above, consider
#
# 1. Whether the algorithm will be able to generate high-probability samples at all;
# 2. Whether the algorithm, across runs, will generate samples from both modes;
# 3. Whether the algorithm, within a single run, will explore both modes.
#
# <hr />
