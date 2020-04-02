# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Cora Julia 1.1.1
#     language: julia
#     name: cora-julia-1.1
# ---

# # Modeling with black-box Julia code 

# This notebook shows how "black-box" code like algorithms and simulators can be included in probabilistic models that are expressed as generative functions.

# We will write a generative probabilistic model of the motion of an intelligent agent that is navigating a two-dimensional scene. The model will be *algorithmic* --- it will invoke a path planning algorithm implemented in regular Julia code to generate the agent's motion plan from its destination its map of the scene.

using Gen

# We also load the `GenViz` Julia package, which is a JavaScript-based visualization framework designed for use with Gen:

using GenViz

# We start a GenViz visualization server that we will use throughout the notebook:

viz_server = VizServer(8090)
sleep(1)

# First, we load some basic geometric primitives for a two-dimensional scene in the following file:

include("../inverse-planning/geometric_primitives.jl");

# This file defines two-dimensional `Point` data type with fields `x` and `y`:

point = Point(1.0, 2.0)
println(point.x)
println(point.y)

# The file also defines an `Obstacle` data type, which represents a polygonal obstacle in a two-dimensional scene, that is constructed from a list of vertices. Here, we construct a square:

obstacle = Obstacle([Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0), Point(1.0, 1.0)]);

# Next, we load the definition of a `Scene` data type that represents a two-dimensional scene.

include("../inverse-planning/scene.jl");

# The scene spans a rectangle of on the two-dimensional x-y plane, and contains a list of obstacles, which is initially empty:
#
# `scene = Scene(xmin::Real, xmax::Real, ymin::Real, ymax::real)`

scene = Scene(0, 1, 0, 1)

# Obstacles are added to the scene with the `add_obstacle!` function:

add_obstacle!(scene, obstacle);

# The file also defines functions that simplify the construction of obstacles:
#
# `make_square(center::Point, size::Float64)` constructs a square-shaped obstacle centered at the given point with the given side length:

obstacle = make_square(Point(0.30, 0.20), 0.1)

# `make_line(vertical::Bool, start::Point, length::Float64, thickness::Float64)` constructs an axis-aligned line (either vertical or horizontal) with given thickness that extends from a given strating point for a certain length:

obstacle = make_line(false, Point(0.20, 0.40), 0.40, 0.02)

# We now construct a scene value that we will use in the rest of the notebook:

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

# We visualize the scene below. Don't worry about the details of this cell. It should display a two-dimensional map of the scene.

info = Dict("scene" => scene)
viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/overlay-viz/dist"), info)
displayInNotebook(viz)

# Next, we load a file that defines a `Path` data type (a sequence of `Points`), and a `plan_path` method, which  uses a path planning algorithm based on rapidly exploring random tree (RRT, [1]) to find a sequence of `Point`s beginning with `start` and ending in `dest` such that the line segment between each consecutive pair of points does not intersect any obstacles in the scene. The planning algorithm may fail to find a valid path, in which case it will return a value of type `Nothing`.
#
# `path::Union{Path,Nothing} = plan_path(start::Point, dest::Point, scene::Scene, planner_params::PlannerParams)`
#
# [1] Rapidly-exploring random trees: A new tool for path planning. S. M. LaValle. TR 98-11, Computer Science Dept., Iowa State University, October 1998,

include("../inverse-planning/planning.jl");

# Let's use `plan_path` to plan a path from the lower-left corner of the scene into the interior of the box.

start = Point(0.1, 0.1)
dest = Point(0.5, 0.5)
planner_params = PlannerParams(300, 3.0, 2000, 1.)
path = plan_path(start, dest, scene, planner_params)

# We visualize the path below. The start location is shown in blue, the destination in red, and the path in orange. Run the cell above followed by the cell below a few times to see the variability in the paths generated by `plan_path` for these inputs.

info = Dict("start"=> start, "dest" => dest, "scene" => scene, "path_edges" => get_edges(path))
viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/overlay-viz/dist"), info)
displayInNotebook(viz)

# We also need a model for how the agent moves along its path.
# We will assume that the agent moves along its path a constant speed. The file loaded above also defines a method (`walk_path`) that computes the locations of the agent at a set of timepoints (sampled at time intervals of `dt` starting at time `0.`), given the path and the speed of the agent:
#
# `locations::Vector{Point} =  walk_path(path::Path, speed::Float64, dt::Float64, num_ticks::Int)`

speed = 1.
dt = 0.1
num_ticks = 10;
locations = walk_path(path, speed, dt, num_ticks)
println(locations)

# Now, we are prepated to write our generative model for the motion of the agent.

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
    speed = @trace(uniform(0, 1), :speed)

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

# We can now perform a traced execution of `agent_model` and print out the random choices it made:

planner_params = PlannerParams(300, 3.0, 2000, 1.)
(trace, _) = Gen.generate(agent_model, (scene, dt, num_ticks, planner_params));
choices = Gen.get_choices(trace)
display(choices)

# Next we explore the assumptions of the model by sampling many traces from the generative function and visualizing them. We have created a visualization specialized for this generative function for use with the `GenViz` package, in the directory `../inverse-planning/grid-viz/dist`. We have also defined a `trace_to_dict` method to convert the trace into a value that can be automatically serialized into a JSON string for use by the visualization:

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
    measurements = Vector{Point}(undef, num_ticks)
    for i=1:num_ticks
        measurements[i] = Point(choices[:meas => (i, :x)], choices[:meas => (i, :y)])
    end
    d["measurements"] = measurements

    return d
end;

# Let's visualize some traces of the function, with the start location fixed to a point in the lower-left corner:

# +
constraints = Gen.choicemap()
constraints[:start_x] = 0.1
constraints[:start_y] = 0.1

viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/grid-viz/dist"), [])
for i=1:12
    (trace, _) = Gen.generate(agent_model, (scene, dt, num_ticks, planner_params), constraints)
    putTrace!(viz, i, trace_to_dict(trace))
end
displayInNotebook(viz)
# -

# In this visualization, the start location is represented by a blue dot, and the destination is represented by a red dot. The measured coordinates at each time point are represented by black dots. The path, if path planning was succesfull, is shown as a gray line from the start point to the destination point. Notice that the speed of the agent is different in each case.

# ### Exercise
#
# Constrain the start and destination points to two particular locations in the scene, and visualize the distribution on paths. Find a start point and destination point such that the distributions on paths is multimodal (i.e. there are two spatially separated paths that the agent may take). Describe the variability in the planned paths.

# ### Solution
#
#

# We now write a simple algorithm for inferring the destination of an agent given (i) the scene, (ii) the start location of the agent, and (iii) a sequence of measured locations of the agent for each tick.
#
# We will assume the agent starts in the lower left-hand corner.

start = Point(0.1, 0.1);

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

# We visualize this data set on top of the scene:

info = Dict("start" => start, "scene" => scene, "measurements" => measurements)
viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/overlay-viz/dist"), info)
displayInNotebook(viz)

# Next, we write a simple inference program for this task:

function do_inference_agent_model(scene::Scene, dt::Float64, num_ticks::Int, planner_params::PlannerParams, start::Point,
                                  measurements::Vector{Point}, amount_of_computation::Int)
    
    # Create an "Assignment" that maps model addresses (:y, i)
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
    (trace, _) = Gen.importance_resampling(agent_model, (scene, dt, num_ticks, planner_params), observations, amount_of_computation)
    
    return trace
end;

# Below, we run this algorithm 1000 times, to generate 1000 approximate samples from the posterior distribution on the destination. The inferred destinations should appear as red dots on the map.

info = Dict("measurements" => measurements, "scene" => scene, "start" => start)
viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/overlay-viz/dist"), info)
openInNotebook(viz)
sleep(5)
for i=1:1000
    trace = do_inference_agent_model(scene, dt, num_ticks, planner_params, start, measurements, 50)
    putTrace!(viz, i, trace_to_dict(trace))
end
displayInNotebook(viz)

# ### Exercise
# The first argument to `PlannerParams` is the number of iterations of the RRT algorithm to use. The third argument to `PlannerParams` is the number of iterations of path refinement. These parameters affect the distribution on paths of the agent. Visualize traces of the `agent_model` for with a couple different settings of these two parameters to the path planning algorithm for fixed starting point and destination point. Try setting them to smaller values. Discuss.

# ### Solution
#
# We have provided starter code.

constraints = Gen.choicemap()
constraints[:start_x] = 0.1
constraints[:start_y] = 0.1;

# Modify the `PlannerParams` in the two cells below.

# +
planner_params = PlannerParams(300, 3.0, 2000, 1.) # < change this line>

viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/grid-viz/dist"), [])
for i=1:12
    (trace, _) = Gen.generate(agent_model, (scene, dt, num_ticks, planner_params), constraints)
    putTrace!(viz, i, trace_to_dict(trace))
end
displayInNotebook(viz)

# +
planner_params = PlannerParams(300, 3.0, 2000, 1.) # < change this line>

viz = Viz(viz_server, joinpath(@__DIR__, "../inverse-planning/grid-viz/dist"), [])
for i=1:12
    (trace, _) = Gen.generate(agent_model, (scene, dt, num_ticks, planner_params), constraints)
    putTrace!(viz, i, trace_to_dict(trace))
end
displayInNotebook(viz)
