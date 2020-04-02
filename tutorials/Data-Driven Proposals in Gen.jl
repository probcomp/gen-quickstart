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

# # Tutorial: Data-Driven Proposals in Gen
#
# This tutorial introduces you to an important inference programming feature in Gen --- using custom "data-driven" proposals to accelerate Monte Carlo inference. Data-driven proposals use information in the observed data set to choose the proposal distibution for latent variables in a generative model. Data-driven proposals can have trainable parameters that are trained data simulated from a generative model to improve their efficacy. This training process is sometimes called 'amortized inference' or 'inference compilation'.
#
# We focus on using data-driven proposals with importance sampling, which is one of the simpler classes of Monte Carlo inference algorithms. Data-driven proposals can also be used with Markov Chain Monte Carlo (MCMC) and sequential Monte Carlo (SMC), but these are not covered in this tutorial.
#
# This tutorial builds on a probabilistic model for the motion of an autonomous agent that was introduced in the [introduction to modeling tutorial](https://github.com/probcomp/gen-examples/tree/master/tutorial-modeling-intro). We show that we can improve the efficiency of inference in this model using two types of custom proposals for the destination of the agent: First, we write a hand-coded data-driven proposal with a single parameter that we tune using amortized inference. Second, we write a data-driven proposal based on a deep neural network, which we also train using amortized inference. We show an implementation of the neural-network based proposal using the built-in modeling language, and then an implementation using the TensorFlow modeling DSL.
#
# Data-driven proposals that are trained on simulated data can be seen from two perspectives: For practitioners of inference in generative models, they can be seen learning a proposal distribution that approximates the conditional distribution on latent variables given observations. Practitioners of supervised machine learning may view data-driven proposals as discriminative models trained on simulated data. By writing the data simulator in a probabilistic programming language, we can embed the probabilistic discriminative model in Monte Carlo algorithms, which can assist with robustness and accuracy. This is an active area of research.
#
# ## Outline
#
# **Section 1.** [Recap on the generative model of an autonomous agent](#model-recap)
#
# **Section 2.** [Writing a data-driven proposal as a generative function](#custom-proposal)
#
# **Section 3.** [Using a data-driven proposal within importance sampling](#using)
#
# **Section 4.** [Training the parameters of a data-driven proposal](#training)
#
# **Section 5.** [Writing and training a deep-learning based data-driven proposal](#deep)
#
# **Section 6.** [Writing a data-driven proposal that uses TensorFlow](#tf)

using Gen
using GenViz
using PyPlot
using JLD

viz_server = VizServer(8092)
sleep(1)

# ## 1: Recap on the generative model of an autonomous agent   <a name="model-recap"></a>

# We begin by loading the source files for the generative model of an autonomous agent that was introduced in a previous tutorial:

include("../inverse-planning/geometric_primitives.jl");
include("../inverse-planning/scene.jl");
include("../inverse-planning/planning.jl");

# We redefine the generative model:

@gen function agent_model(scene::Scene, dt::Float64, num_ticks::Int, planner_params::PlannerParams)

    # sample the start point of the agent from the prior
    start_x = @trace(uniform(0, 1), :start_x)
    start_y = @trace(uniform(0, 1), :start_y)
    start = Point(start_x, start_y)

    # sample the destination point of the agent from the prior
    dest_x = @trace(uniform(0, 1), :dest_x)
    dest_y = @trace(uniform(0, 1), :dest_y)
    dest = Point(dest_x, dest_y)

    # plan a path that avoids obstacles in the scene
    maybe_path = plan_path(start, dest, scene, planner_params)
    planning_failed = maybe_path == nothing
    
    # sample the speed from the prior
    speed = @trace(uniform(0.3, 1), :speed)

    if planning_failed
        
        # path planning failed, assume the agent stays as the start location indefinitely
        locations = fill(start, num_ticks)
    else
        
        # path planning succeeded, move along the path at constant speed
        locations = walk_path(maybe_path, speed, dt, num_ticks)
    end

    # generate noisy measurements of the agent's location at each time point
    noise = 0.01
    for (i, point) in enumerate(locations)
        x = @trace(normal(point.x, noise), :meas => (i, :x))
        y = @trace(normal(point.y, noise), :meas => (i, :y))
    end

    return (planning_failed, maybe_path)
end;

# And we redefine a function that converts a trace of this model into a value that is easily serializable to JSON, for use with the GenViz visualization framework:

function trace_to_dict(trace)
    args = Gen.get_args(trace)
    (scene, dt, num_ticks, planner_params) = args
    choices = Gen.get_choices(trace)
    (planning_failed, maybe_path) = Gen.get_retval(trace)

    d = Dict()

    # scene (the obstacles)
    d["scene"] = scene

    # the points along the planned path
    if planning_failed
        d["path"] = []
    else
        d["path"] = maybe_path.points
    end

    # start and destination location
    d["start"] = Point(choices[:start_x], choices[:start_y])
    d["dest"] = Point(choices[:dest_x], choices[:dest_y])

    # the observed location of the agent over time
    local measurements
    measurements = Vector{Point}(undef, num_ticks)
    for i=1:num_ticks
        measurements[i] = Point(choices[:meas => (i, :x)], choices[:meas => (i, :y)])
    end
    d["measurements"] = measurements

    return d
end;

# We redefine a scene:

scene = Scene(0, 1, 0, 1)
add_obstacle!(scene, make_square(Point(0.30, 0.20), 0.1))
add_obstacle!(scene, make_square(Point(0.83, 0.80), 0.1))
add_obstacle!(scene, make_square(Point(0.80, 0.40), 0.1))
horizontal = false
vertical = true
wall_thickness = 0.02
add_obstacle!(scene, make_line(horizontal, Point(0.20, 0.40), 0.40, wall_thickness))
add_obstacle!(scene, make_line(vertical, Point(0.60, 0.40), 0.40, wall_thickness))
add_obstacle!(scene, make_line(horizontal, Point(0.60 - 0.15, 0.80), 0.15 + wall_thickness, wall_thickness))
add_obstacle!(scene, make_line(horizontal, Point(0.20, 0.80), 0.15, wall_thickness))
add_obstacle!(scene, make_line(vertical, Point(0.20, 0.40), 0.40, wall_thickness));

# We will assume the agent starts in the lower left-hand corner. And we will assume particular parameters for the planning algorithm that the agent uses. We will also assume that there are 10 measurements, separated `0.1` time units.

start = Point(0.1, 0.1)
dt = 0.1
num_ticks = 10
planner_params = PlannerParams(300, 3.0, 2000, 1.);

# We will infer the destination of the agent for the given sequence of observed locations:

measurements = [
    Point(0.0980245, 0.104775),
    Point(0.113734, 0.150773),
    Point(0.100412, 0.195499),
    Point(0.114794, 0.237386),
    Point(0.0957668, 0.277711),
    Point(0.140181, 0.31304),
    Point(0.124384, 0.356242),
    Point(0.122272, 0.414463),
    Point(0.124597, 0.462056),
    Point(0.126227, 0.498338)];

# We redefine the generic importance sampling algorithm that we used in the previous notebook:

function do_inference(scene::Scene, dt::Float64, num_ticks::Int, planner_params::PlannerParams, start::Point,
                      measurements::Vector{Point}, amount_of_computation::Int)
    
    # Create a choice map that maps model addresses (:y, i)
    # to observed values ys[i]. We leave :slope and :intercept
    # unconstrained, because we want them to be inferred.
    observations = Gen.choicemap()
    observations[:start_x] = start.x
    observations[:start_y] = start.y
    for (i, m) in enumerate(measurements)
        observations[:meas => (i, :x)] = m.x
        observations[:meas => (i, :y)] = m.y
    end
    
    # Call importance_resampling to obtain a likely trace consistent
    # with our observations.
    (trace, _) = Gen.importance_resampling(agent_model, (scene, dt, num_ticks, planner_params),
        observations, amount_of_computation)
    
    return trace
end;

# Below, we run this algorithm 1000 times, to generate 1000 approximate samples from the posterior distribution on the destination. The inferred destinations should appear as red dots on the map. First, we abstract this into a function.

function visualize_inference(measurements, scene, start; computation_amt=50, samples=1000)
    info = Dict("measurements" => measurements, "scene" => scene, "start" => start)
    viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/overlay-viz/dist"), info)
    openInNotebook(viz)
    sleep(5)
    for i=1:samples
        trace = do_inference(scene, dt, num_ticks, planner_params, start, measurements, computation_amt)
        putTrace!(viz, i, trace_to_dict(trace))
    end
    displayInNotebook(viz)
end;

visualize_inference(measurements, scene, start, computation_amt=50, samples=1000)

# ## 2. Writing a data-driven proposal as a generative function <a name="custom-proposal"></a>

# The inference algorithm above used a variant of [`Gen.importance_resampling`](https://probcomp.github.io/Gen/dev/ref/inference/#Gen.importance_resampling) that does not take a custom proposal distribution. It uses the default proposal distribution associated with the generative model. For generative functions defined using the built-in modeling DSL, the default proposal distribution is based on *ancestral sampling*, which involves sampling unconstrained random choices from the distributions specified in the generative mode.

# We can sample from the default proposal distribution using `Gen.initialize`. The cell below shows samples of the destination from this distribution.

info = Dict("measurements" => measurements, "scene" => scene, "start" => start)
viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/overlay-viz/dist"), info)
for i=1:1000
    (trace, _) = Gen.generate(agent_model, (scene, dt, num_ticks, planner_params))
    putTrace!(viz, i, trace_to_dict(trace))
end
displayInNotebook(viz)

# Intuitively, if we see the data set above (where blue is the starting location), we might guess that the agent is more likely to be heading into the upper part of the scene. This is because we don't expect the agent to unecessarily backtrack on its route to its destnation. A simple heuristic for biasing the proposal distribution of the destination using just the first measurement and the last measurement might be:
#
# - If the x-coordinate of the last measurement is greater than the x-coordinate of the first measurement, we think the agent is probably headed generally to the right. Make values `:dest_x` that are greater than the x-coordinate of the last measurement more probable.
#
# - If the x-coordinate of the last measurment is less than the x-coordinate of the first measurement, we think the agent is probably headed generally to the left. Make values  `:dest_x` that are smaller than the x-coordinate of the last measurement more probable.
#
# We can apply the same heuristic separately for the y-coordinate.

# To implement this idea, we discretize the x-axis and y-axis of the scene into bins:

num_x_bins = 5
num_y_bins = 5;

# We will propose the x-coordinate of the destination from a [piecewise_uniform](https://probcomp.github.io/Gen/dev/ref/distributions/#Gen.piecewise_uniform) distribution, where we set higher probability for certain bins based on the heuristic described above and use a uniform continuous distribution for the coordinate within a bin. The `compute_bin_probs` function below computes the probability for each bin. The bounds of the scene are given by `min` and `max`. The coordinates of the first and last measured points respectively are given by `first` and `last`. We compute the probability by assigning a "score" to each bin based on the heuristic above --- if the bin should receive lower probability, it gets a score of 1., and if it should receive higher probability, it gets a bin of `score_high`, where `score_high` is some value greater than 1.

# +
function compute_bin_prob(first::Float64, last::Float64, bin::Int, last_bin::Int, score_high)
    last >= first && bin >= last_bin && return score_high
    last < first && bin <= last_bin && return score_high
    return 1.
end

function compute_bin_probs(num_bins::Int, min::Float64, max::Float64, first::Float64, last::Float64, score_high)
    bin_len = (max - min) / num_bins
    last_bin = Int(floor(last / bin_len)) + 1
    probs = [compute_bin_prob(first, last, bin, last_bin, score_high) for bin=1:num_bins]
    total = sum(probs)
    return [p / total for p in probs]
end;
# -

# We will see how to automatically tune the value of `score_high` shortly. For now, we will use a value of 5. Below, we see that for the data set of measurements, shown above the probabilities of higher bins are indeed 5x larger than those of lower bins, becuase the agent seems to be headed up. 

compute_bin_probs(num_y_bins, scene.ymin, scene.ymax, measurements[1].y, measurements[end].y, 5.)

# Next, we write a generative function that applies this heuristic for both the x-coordinate and y-coordinate, and samples the destination coordinates at addresses `:dest_x` and `:dest_y`.

@gen function custom_dest_proposal(measurements::Vector{Point}, scene::Scene)

    score_high = 5.
    
    x_first = measurements[1].x
    x_last = measurements[end].x
    y_first = measurements[1].y
    y_last = measurements[end].y
    
    # sample dest_x
    x_probs = compute_bin_probs(num_x_bins, scene.xmin, scene.xmax, x_first, x_last, score_high)
    x_bounds = collect(range(scene.xmin, stop=scene.xmax, length=num_x_bins+1))
    @trace(Gen.piecewise_uniform(x_bounds, x_probs), :dest_x)
    
    # sample dest_y
    y_probs = compute_bin_probs(num_y_bins, scene.ymin, scene.ymax, y_first, y_last, score_high)
    y_bounds = collect(range(scene.ymin, stop=scene.ymax, length=num_y_bins+1))
    @trace(Gen.piecewise_uniform(y_bounds, y_probs), :dest_y)
    
    return nothing
end;

# We can propose values of random choices from the proposal function using [`Gen.propose`](https://probcomp.github.io/Gen/dev/ref/gfi/#Gen.propose). This method returns the choices, as well as some other information, which we won't need for our purposes. For now, you can think of `Gen.propose` as similar to `Gen.initialize` except that it does not produce a full execution trace (only the choices), and it does not accept constraints. We can see the random choices for one run of the proposal on our data set:

(proposed_choices, _, _) = Gen.propose(custom_dest_proposal, (measurements, scene))
display(proposed_choices)

# The function below runs the proposal 1000 times. For each run, it generates a trace of the model where the `:dest_x` and `:dest_y` choices are constrained to the proposed values, and then visualizes the resulting traces. We make the proposal a parameter of the function because we will be visualizing the output distribution of various proposals later in the notebook.

function visualize_custom_destination_proposal(measurements, start, dest_proposal; num_samples=100)
    info = Dict("measurements" => measurements, "scene" => scene, "start" => start)
    viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/overlay-viz/dist"), info)
    for i=1:num_samples
        (proposed_choices, _) = Gen.propose(dest_proposal, (measurements, scene))
        (trace, _) = Gen.generate(agent_model, (scene, dt, num_ticks, planner_params), proposed_choices)
        putTrace!(viz, i, trace_to_dict(trace))
    end
    displayInNotebook(viz)    
end;

# Let's visualize the output distribution of `custom_dest_proposal` for our data set:

visualize_custom_destination_proposal(measurements, start, custom_dest_proposal, num_samples=1000)

# We see that the proposal distribution indeed samples destinations in the top half of the scene with higher probability than destinations in the bottom half.

# ## 3. Using a data-driven proposal within importance sampling <a name="using"></a>

# We now use our data-driven proposal within an inference algorithm. There is a second variant of [Gen.importance_resampling](https://probcomp.github.io/Gen/dev/ref/inference/#Gen.importance_resampling) that accepts a generative function representing a custom proposal. This proposal generative function makes traced random choices at the addresses of a subset of the unobserved random choices made by the generative model. In our case, these addresses are `:dest_x` and `:dest_y`. Below, we write an inference program that uses this second variant of importance resampling. Because we will experiment with different data-driven proposals, we make the proposal into an agument of our inference program. We assume that the proposal accepts arguments `(measurements, scene)`.

function do_inference_data_driven(dest_proposal::GenerativeFunction,
                                              scene::Scene, dt::Float64,
                                              num_ticks::Int, planner_params::PlannerParams,
                                              start::Point, measurements::Vector{Point}, 
                                              amount_of_computation::Int)
    
    observations = Gen.choicemap((:start_x, start.x), (:start_y, start.y))
    for (i, m) in enumerate(measurements)
        observations[:meas => (i, :x)] = m.x
        observations[:meas => (i, :y)] = m.y
    end
    
    # invoke the variant of importance_resampling that accepts a custom proposal (dest_proposal)
    # the arguments to the custom proposal are (measurements, scene)
    (trace, _) = Gen.importance_resampling(agent_model, (scene, dt, num_ticks, planner_params), observations, 
        dest_proposal, (measurements, scene), amount_of_computation)
    
    return trace
end;

# We also write a function below that runs this algorithm a number of times, and visualizes the result:

function visualize_data_driven_inference(measurements, scene, start, proposal; amt_computation=50, samples=1000)
    info = Dict("measurements" => measurements, "scene" => scene, "start" => start)
    viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/overlay-viz/dist"), info)
    openInNotebook(viz)
    sleep(5)
    for i=1:samples
        trace = do_inference_data_driven(proposal, 
            scene, dt, num_ticks, planner_params, start, measurements, amt_computation)
        putTrace!(viz, i, trace_to_dict(trace))
    end
    displayInNotebook(viz)
end;

# We visualize the results with the amount of computation set fo `5`:

visualize_data_driven_inference(measurements, scene, start, custom_dest_proposal, amt_computation=5, samples=1000)

# We compare this to the original algorithm that used the default proposal, for the same "amount of computation" of 5.

visualize_inference(measurements, scene, start, computation_amt=5, samples=1000)

# We see that the results are somewhat more accurate using the data-driven proposal.  In particular, there is less probability mass in the lower left corner when using the data-driven proposal.

# ## 4. Training the parameters of a data-driven proposal <a name="training"></a>

# Our choice of the `score_high` value of 5. was somewhat arbitrary. To use more informed value, we can make `score_high` into a [*trainable parameter*](https://probcomp.github.io/Gen/dev/ref/modeling/#Trainable-parameters-1) of the generative function. Below, we write a new version of the proposal function that makes `score_high` trainable. However, the optimization algorithms we will use for training work best with *unconstrained* parameters (parameters that can take any value on the real line), but `score_high` must be positive. Therefore, we introduce an unconstrained trainable parameter mamed `log_score_high`, and use `exp()` to ensure that `score_high` is positive:

@gen function custom_dest_proposal_trainable(measurements::Vector{Point}, scene::Scene)

    @param log_score_high::Float64
    
    x_first = measurements[1].x
    x_last = measurements[end].x
    y_first = measurements[1].y
    y_last = measurements[end].y
    
    # sample dest_x
    x_probs = compute_bin_probs(num_x_bins, scene.xmin, scene.xmax, x_first, x_last, exp(log_score_high))
    x_bounds = collect(range(scene.xmin, stop=scene.xmax, length=num_x_bins+1))
    @trace(Gen.piecewise_uniform(x_bounds, x_probs), :dest_x)
    
    # sample dest_y
    y_probs = compute_bin_probs(num_y_bins, scene.ymin, scene.ymax, y_first, y_last, exp(log_score_high))
    y_bounds = collect(range(scene.ymin, stop=scene.ymax, length=num_y_bins+1))
    @trace(Gen.piecewise_uniform(y_bounds, y_probs), :dest_y)
    
    return nothing
end;

# We initialize the value of `score_high` to 1. For this value, our custom proposal gives a uniform distribution, and is the same as the default proposal.

Gen.init_param!(custom_dest_proposal_trainable, :log_score_high, 0.);

# Let's visualize the proposed distribution prior to training to confirm that it is a uniform distribution.

visualize_custom_destination_proposal(measurements, start, custom_dest_proposal_trainable, num_samples=1000)

# Now, we train the generative function. First, we will require a data-generator that generates the training data. The data-generator is a function of no arguments that returns a tuple of the form `(inputs, constraints)`. The `inputs` are the arguments to the generative function being trained, and the `constraints` contains the desired values of random choices made by the function for those arguments. For the training distribution, we will use the distribution induced by the generative model (`agent_model`), restricted to cases where planning actually succeeded. When planning failed, the agent just stays at the same location for all time, and we won't worry about tuning our proposal for that case. The training procedure will attempt to maximize the expected conditional log probablity (density) that the proposal function generates the constrained values, when run on the arguments. Note that this is an *average case* objective function --- the resulting proposal distribution may perform better on some data sets than others.

function data_generator()
    
    # since these names are used in the global scope, explicitly declare it
    # local to avoid overwriting the global variable
    local measurements
    local choices
    
    # obtain an execution of the model where planning succeeded
    done = false
    while !done
        (choices, _, retval) = Gen.propose(agent_model, (scene, dt, num_ticks, planner_params))
        (planning_failed, maybe_path) = retval       
        done = !planning_failed
    end

    # construct arguments to the proposal function being trained
    measurements = [Point(choices[:meas => (i, :x)], choices[:meas => (i, :y)]) for i=1:num_ticks]
    inputs = (measurements, scene)
    
    # construct constraints for the proposal function being trained
    constraints = Gen.choicemap()
    constraints[:dest_x] = choices[:dest_x]
    constraints[:dest_y] = choices[:dest_y]
    
    return (inputs, constraints)
end;

# Next, we choose type of optimization algorithm we will use for training. Gen supports a set of gradient-based optimization algorithms (see [Optimizing Trainable Parameters](https://probcomp.github.io/Gen/dev/ref/parameter_optimization/#Optimizing-Trainable-Parameters-1)). Here we will use gradient descent with a fixed step size of 0.001.

update = Gen.ParamUpdate(Gen.FixedStepGradientDescent(0.001), custom_dest_proposal_trainable);

# Finally, we use the [`Gen.train!`](https://probcomp.github.io/Gen/dev/ref/inference/#Gen.train!) method to actually do the training.
#
# For each epoch, `Gen.train!` makes `epoch_size` calls to the data-generator to construct a batch of training data for that epoch. Then, it iteratively selects `num_minibatch` subsets of the epoch training data, each of size `100`, and applies the update once per minibatch. At the end of the epoch, it generates another batch of evaluation data (of size `evaluation_size`) which it uses to estimate the objective function (the expected conditional log likelihood under the data-generating distribution).
#
# Here, we are running 200 gradient-descent updates, where each update is using a gradient estimate obtained from 100 training examples. The method prints the estimate of the objective function after each epoch.

@time scores = Gen.train!(custom_dest_proposal_trainable, data_generator, update,
    num_epoch=200, epoch_size=100, num_minibatch=1, minibatch_size=100, evaluation_size=100, verbose=true);

plot(scores)
xlabel("Iterations of stochastic gradient descent")
ylabel("Estimate of expected conditional log probability density");

# We can read out the new value for `score_high`:

println(exp(Gen.get_param(custom_dest_proposal_trainable, :log_score_high)))

# We see that the optimal value of the parameter is indeed larger than our initial guess. This validates that the heuristic is indeed a useful one. We visualize the proposal distribution below:

visualize_custom_destination_proposal(measurements, start, custom_dest_proposal_trainable, num_samples=1000)

# We can visualize the results of inference, using this newly trained proposal:

visualize_data_driven_inference(measurements, scene, start, custom_dest_proposal_trainable,
    amt_computation=5, samples=1000)

# ------------
# ### Exercise
#
# Can you devise a data-driven proposal for the speed of the agent? Do you expect it to work equally well on all data sets? You do not need to implement it.

# ### Solution

# ----------------

# ## 5. Writing and training a deep learning based data-driven proposal <a name="deep"></a>

# The heuristic data-driven proposal above gave some improvement in efficiency, but it was very simple. One way of constructing complex data-driven proposals is to parametrize the proposal with a deep neural network or use another class of high-capacity machine learning model (e.g. random forest). Here, we will will write a data-driven proposal for the destination of the agent that uses deep neural networks.

# First, we define a sigmoid function for the nonlinearity in our networks.

nonlinearity(x) = 1.7159 * tanh.(x * 0.66666);

# We will use a deep neural network with two hidden layers that takes as input x- and y- coordinates of the first and last measurement (4 values) and produces as output a vector of un-normalized probabilities, one for each bin of the x-dimension. We will later sample `:dest_x` from this distribution.

function dest_x_neural_net(nn_params, x_first::Real, y_first::Real, x_last::Real, y_last::Real)
    (W1, b1, W2, b2, W3, b3) = nn_params
    input_layer = [x_first, y_first, x_last, y_last]
    hidden_layer_1 = nonlinearity(W1 * input_layer .+ b1)
    hidden_layer_2 = nonlinearity(W2 * hidden_layer_1 .+ b2)
    output_layer = exp.(W3 * hidden_layer_2 .+ b3)
    return output_layer
end;

# After sampling the value of `:dest_x`, we will use a second deep neural network that takes as input the same four values extracted from the measurements, as well as the sampled value of `:dest_x` (a total of 5 inputs), and produces a vector of un-normalized probabilities, one for each bin of the y-dimension. We will sample `:dest_y` from this distribution.

function dest_y_neural_net(nn_params, x_first::Real, y_first::Real, x_last::Real, y_last::Real)#, dest_x::Real)
    (W1, b1, W2, b2, W3, b3) = nn_params
    input_layer = [x_first, y_first, x_last, y_last]#, dest_x]
    hidden_layer_1 = nonlinearity(W1 * input_layer .+ b1)
    hidden_layer_2 = nonlinearity(W2 * hidden_layer_1 .+ b2)
    output_layer = exp.(W3 * hidden_layer_2 .+ b3)
    return output_layer
end;

# Now that we have defined our neural networks, we define our new proposal. This generative function has a number of parameters.

# +
scale_coord(coord, min, max) = (coord / (max - min)) - 0.5

@gen function custom_dest_proposal_neural(measurements::Vector{Point}, scene::Scene)
        
    @param x_W1::Matrix{Float64}
    @param x_b1::Vector{Float64}
    @param x_W2::Matrix{Float64}
    @param x_b2::Vector{Float64}
    @param x_W3::Matrix{Float64}
    @param x_b3::Vector{Float64}
    
    @param y_W1::Matrix{Float64}
    @param y_b1::Vector{Float64}
    @param y_W2::Matrix{Float64}
    @param y_b2::Vector{Float64}
    @param y_W3::Matrix{Float64}
    @param y_b3::Vector{Float64}
    
    num_x_bins = length(x_b3)
    num_y_bins = length(y_b3)
    
    # scale inputs to be in the range [-0.5, 0.5]
    x_first = scale_coord(measurements[1].x, scene.xmin, scene.xmax)
    x_last = scale_coord(measurements[end].x, scene.xmin, scene.xmax)
    y_first = scale_coord(measurements[1].y, scene.ymin, scene.ymax)
    y_last = scale_coord(measurements[end].y, scene.ymin, scene.ymax)
    
    # sample dest_x
    x_bounds = collect(range(scene.xmin, stop=scene.xmax, length=num_x_bins+1))
    x_probs = dest_x_neural_net((x_W1, x_b1, x_W2, x_b2, x_W3, x_b3), x_first, y_first, x_last, y_last)
    @trace(Gen.piecewise_uniform(x_bounds, x_probs / sum(x_probs)), :dest_x)
    
    # sample dest_y
    y_bounds = collect(range(scene.xmin, stop=scene.xmax, length=num_y_bins+1))
    y_probs = dest_y_neural_net((y_W1, y_b1, y_W2, y_b2, y_W3, y_b3), x_first, y_first, x_last, y_last)
    @trace(Gen.piecewise_uniform(y_bounds, y_probs / sum(y_probs)), :dest_y)
    
    return nothing
end;
# -

# We will use 50 hidden units in each of the layers of the two networks:

num_hidden_1 = 50
num_hidden_2 = 50;

# Next, we initialize the parameters:

# +
import Random
Random.seed!(3)

init_weight(shape...) = (1. / sqrt(shape[2])) * randn(shape...)

init_x_W1 = init_weight(num_hidden_1, 4)
init_x_W2 = init_weight(num_hidden_2, num_hidden_1)
init_x_W3 = init_weight(num_x_bins, num_hidden_2)

# set parameters for dest_x_neural_net predictor network
init_param!(custom_dest_proposal_neural, :x_W1, init_x_W1)
init_param!(custom_dest_proposal_neural, :x_b1, zeros(num_hidden_1))
init_param!(custom_dest_proposal_neural, :x_W2, init_x_W2)
init_param!(custom_dest_proposal_neural, :x_b2, zeros(num_hidden_2))
init_param!(custom_dest_proposal_neural, :x_W3, init_x_W3)
init_param!(custom_dest_proposal_neural, :x_b3, zeros(num_x_bins))

init_y_W1 = init_weight(num_hidden_1, 4)
init_y_W2 = init_weight(num_hidden_2, num_hidden_1)
init_y_W3 = init_weight(num_x_bins, num_hidden_2)

# set parameters for dest_y_neural_net predictor network
init_param!(custom_dest_proposal_neural, :y_W1, init_y_W1)
init_param!(custom_dest_proposal_neural, :y_b1, zeros(num_hidden_1))
init_param!(custom_dest_proposal_neural, :y_W2, init_y_W2)
init_param!(custom_dest_proposal_neural, :y_b2, zeros(num_hidden_2))
init_param!(custom_dest_proposal_neural, :y_W3, init_y_W3)
init_param!(custom_dest_proposal_neural, :y_b3, zeros(num_y_bins));
# -

# Now, we visualize the proposal distribution prior to training:

visualize_custom_destination_proposal(measurements, start, custom_dest_proposal_neural, num_samples=1000)

# It looks like the initial distribution is roughly uniform, like the default proposal.

# Now we train the network stochastic gradient descent with a fixed step size of 0.001 that is shared among all of the trainable parameters.

update = Gen.ParamUpdate(Gen.FixedStepGradientDescent(0.001), custom_dest_proposal_neural);

# We use 100 epochs of training. In each epoch, we generate 1000 training examples, and we apply 100 gradient updates, where each update is based on the gradient estimate obtained from a random set of 100 of the trainable examples. At the end of each epoch, we estimate the objective function value using 10000 freshly sampled examples. This process takes about 10 minutes to run, so we have precomputed the results for you
#
# ```julia
# @time scores = Gen.train!(custom_dest_proposal_neural, data_generator, update,
#     num_epoch=100, epoch_size=1000, num_minibatch=100, minibatch_size=100,
#     evaluation_size=1000, verbose=true);
#     
# let data = Dict()
#     for name in [:x_W1, :x_b1, :x_W2, :x_b2, :x_W3, :x_b3, :y_W1, :y_b1, :y_W2, :y_b2, :y_W3, :y_b3]
#         data[(:param, name)] = Gen.get_param(custom_dest_proposal_neural, name)
#     end
#     data[:scores] = scores
#     save("params/custom_dest_proposal_neural_trained.jld", "data", data)
# end
# ```

# We load the results here:

scores = let data = load("params/custom_dest_proposal_neural_trained.jld", "data")
    for name in [:x_W1, :x_b1, :x_W2, :x_b2, :x_W3, :x_b3, :y_W1, :y_b1, :y_W2, :y_b2, :y_W3, :y_b3]
        Gen.init_param!(custom_dest_proposal_neural, name, data[(:param, name)])
    end
    data[:scores]
end;

# We plot the estimate of the objective function over epochs:

plot(scores)
xlabel("Epochs")
ylabel("Estimate of the expected conditional log probability density");

# Below, we visualize the trained proposal distribution for our data set:

visualize_custom_destination_proposal(measurements, start, custom_dest_proposal_neural, num_samples=1000)

# If we run inference with `amt_computation` set to 5, we see that the inferred distribution reflects the bias of the proposal:

visualize_data_driven_inference(measurements, scene, start, custom_dest_proposal_neural,
    amt_computation=5, samples=1000)

# As we increase the amount of computation, the effect of the proposal's bias is reduced:

visualize_data_driven_inference(measurements, scene, start, custom_dest_proposal_neural,
    amt_computation=50, samples=1000)

# In the limit of infinite computation, the distribution converges to the posterior:

# ------
#
# ### Exercise
#
# Visualize the trained proposal distribution `custom_dest_proposal_neural` on some other data sets, either generated by simulating from the model, or constructed manually. Note that the proposal only reads from the first and last of the measurements. Discuss the results.

# ### Solution

# ------------

# ---------
# ### Exercise
#
# What is the objective value for the default proposal, which is the uniform distribution on the entire scene area? Recall that the objective function is the expected conditional log probability density of the proposal evaluated on destination points drawn by simulating from the model.
#
# Hint: The log probability density of the uniform distribution is a constant.

# ----
#
# ### Solution

# ## 6. Writing a data-driven proposal that uses TensorFlow <a name="tf"></a>
#
# The data-driven neural proposal above used Julia code to implement the neural network. It is also possible to implement the neural networks in TensorFlow using the [GenTF](https://github.com/probcomp/GenTF) package that was introduced in a previous tutorial. We load GenTF, [PyCall](https://github.com/JuliaPy/PyCall.jl), and we import the TensorFlow Python module: 

using GenTF
using PyCall
tf = pyimport("tensorflow")
tf.compat.v1.disable_eager_execution()

# Now, we implement the neural network that generate sthe parameters of the distribution of `:dest_x` as a `GenTF.TFFunction`, which is a type of generative function that is constructed from a TensorFlow computation graph.

# +
vec_to_mat(vec) = tf.expand_dims(vec, axis=1)
mat_to_vec(mat) = tf.squeeze(mat, axis=1)

c1 = tf.compat.v1.constant(1.7159, dtype=tf.float64)
c2 = tf.compat.v1.constant(0.6666, dtype=tf.float64)
tf_nonlinearity(val) = tf.scalar_mul(c1, tf.tanh(tf.scalar_mul(c2, val)))

# use TensorFlow Python API to construct computation graph for the dest_x neural network
x_W1 = tf.compat.v1.get_variable("x_W1", dtype=tf.float64, initializer=init_x_W1)
x_b1 = tf.compat.v1.get_variable("x_b1", dtype=tf.float64, initializer=zeros(num_hidden_1))
x_W2 = tf.compat.v1.get_variable("x_W2", dtype=tf.float64, initializer=init_x_W2)
x_b2 = tf.compat.v1.get_variable("x_b2", dtype=tf.float64, initializer=zeros(num_hidden_2))
x_W3 = tf.compat.v1.get_variable("x_W3", dtype=tf.float64, initializer=init_x_W3)
x_b3 = tf.compat.v1.get_variable("x_b3", dtype=tf.float64, initializer=zeros(num_x_bins))
x_nn_input = tf.compat.v1.placeholder(dtype=tf.float64, shape=(4,))
x_nn_hidden_1 = tf_nonlinearity(tf.add(mat_to_vec(tf.matmul(x_W1, vec_to_mat(x_nn_input))), x_b1))
x_nn_hidden_2 = tf_nonlinearity(tf.add(mat_to_vec(tf.matmul(x_W2, vec_to_mat(x_nn_hidden_1))), x_b2))
x_nn_output = tf.exp(tf.add(mat_to_vec(tf.matmul(x_W3, vec_to_mat(x_nn_hidden_2))), x_b3))

# construct a TFFunction generative function for the x neural network
x_nn = GenTF.TFFunction([x_W1, x_b1, x_W2, x_b2, x_W3, x_b3], [x_nn_input], x_nn_output);
# -

# We do the same for the neural network that generates the y-coordinates:

# +
# use TensorFlow Python API to construct computation graph for the dest_y neural network
y_W1 = tf.compat.v1.get_variable("y_W1", dtype=tf.float64, initializer=init_y_W1)
y_b1 = tf.compat.v1.get_variable("y_b1", dtype=tf.float64, initializer=zeros(num_hidden_1))
y_W2 = tf.compat.v1.get_variable("y_W2", dtype=tf.float64, initializer=init_y_W2)
y_b2 = tf.compat.v1.get_variable("y_b2", dtype=tf.float64, initializer=zeros(num_hidden_2))
y_W3 = tf.compat.v1.get_variable("y_W3", dtype=tf.float64, initializer=init_y_W3)
y_b3 = tf.compat.v1.get_variable("y_b3", dtype=tf.float64, initializer=zeros(num_y_bins))
y_nn_input = tf.compat.v1.placeholder(dtype=tf.float64, shape=(4,))
y_nn_hidden_1 = tf_nonlinearity(tf.add(mat_to_vec(tf.matmul(y_W1, vec_to_mat(y_nn_input))), y_b1))
y_nn_hidden_2 = tf_nonlinearity(tf.add(mat_to_vec(tf.matmul(y_W2, vec_to_mat(y_nn_hidden_1))), y_b2))
y_nn_output = tf.exp(tf.add(mat_to_vec(tf.matmul(y_W3, vec_to_mat(y_nn_hidden_2))), y_b3))

# construct a TFFunction generative function for the y neural network
y_nn = GenTF.TFFunction([y_W1, y_b1, y_W2, y_b2, y_W3, y_b3], [y_nn_input], y_nn_output);
# -

# Next, we define a `@gen` function that invokes the two neural network `TFFunction`s and actually samples the proposed values.

@gen function custom_dest_proposal_tf(measurements::Vector{Point}, scene::Scene)
        
    # scale inputs to be in the range [-0.5, 0.5]
    x_first = scale_coord(measurements[1].x, scene.xmin, scene.xmax)
    x_last = scale_coord(measurements[end].x, scene.xmin, scene.xmax)
    y_first = scale_coord(measurements[1].y, scene.ymin, scene.ymax)
    y_last = scale_coord(measurements[end].y, scene.ymin, scene.ymax)
    
    # sample dest_x
    x_bounds = collect(range(scene.xmin, stop=scene.xmax, length=num_x_bins+1))
    x_probs = @trace(x_nn([x_first, y_first, x_last, y_last]), :x_net)
    dest_x = @trace(Gen.piecewise_uniform(x_bounds, x_probs / sum(x_probs)), :dest_x)
    
    # sample dest_y
    y_bounds = collect(range(scene.ymin, stop=scene.ymax, length=num_y_bins+1))
    y_probs = @trace(y_nn([x_first, y_first, x_last, y_last]), :y_net)
    @trace(Gen.piecewise_uniform(y_bounds, y_probs / sum(y_probs)), :dest_y)
    
    return nothing
end;

# To train the proposal, we construct an update that applies a fixed step size gradient descent move. We indicate that we want the update to apply to all the trainable parameters of `x_nn` and all the trainable parameters of `y_nn`. Note that `custom_dest_proposal_tf` does not have any trainable parameters of its own, unlike `custom_dest_proposal_neural`.

update = Gen.ParamUpdate(Gen.FixedStepGradientDescent(0.001),
    x_nn => collect(get_params(x_nn)), y_nn => collect(get_params(y_nn)));

# The training takes about 10 minutes to run on a CPU. We have already run the training code, and saved the parameters:
#
# ```julia
# Gen.train!(custom_dest_proposal_tf, data_generator, update,
#     num_epoch=100, epoch_size=1000, num_minibatch=100, minibatch_size=100,
#     evaluation_size=10, eval_period=100, verbose=true);
#    
# # save state of parameters for x_nn_vec
# saver = tf.compat.v1.train.Saver(Dict(string(var.name) => var for var in Gen.get_params(x_nn_vec)))
# saver.save(GenTF.get_session(x_nn_vec), "params/x_nn.ckpt")
#
# # save state of parameters for y_nn_vec
# saver = tf.compat.v1.train.Saver(Dict(string(var.name) => var for var in Gen.get_params(x_nn_vec)))
# saver.save(GenTF.get_session(x_nn_vec), "params/x_nn.ckpt")
# ```

# We load the parameters below:

# +
saver = tf.compat.v1.train.Saver(Dict(string(var.name) => var for var in Gen.get_params(x_nn)))
saver.restore(GenTF.get_session(x_nn), "params/x_nn.ckpt")

saver = tf.compat.v1.train.Saver(Dict(string(var.name) => var for var in Gen.get_params(y_nn)))
saver.restore(GenTF.get_session(y_nn), "params/y_nn.ckpt")
# -

# We visualize the distribution after training:

visualize_custom_destination_proposal(measurements, start, custom_dest_proposal_tf, num_samples=1000)

# --------
#
# ### Exercise
#
# The training procedure for the neural networks above was not vectorized across training examples. For fast training on a GPU it is important to vectorize the evaluation of gradients across multiple training examples. Write a vectorized version of the `custom_dest_proposal_tf` that takes a scene and a vector of data sets (measurement vectors), and samples a destination point for each of the input data sets. Train it and visualize the proposal distribution, and the results of importance resampling inference that using the proposal for some amount of coputation, for our example data set.
#
# Hint:
#
# - Construct a vectorized version of each of the neural networks that operate on an extra 'training example' dimension.
#
# - Construct a vectorized version of the proposal. It should accept a vector of measurement vectors as one of its arguments. This vectorized propsal should make 2N random choices where N is the batch size.
#
# - Construct a vectorized version of the data generator. It should generate constraints for all random choices of the vectorized proposal.
#
# - Construct a non-vectorized version of the proposal that invokes the vectorized neural networks on a single data set.

# ### Solution
