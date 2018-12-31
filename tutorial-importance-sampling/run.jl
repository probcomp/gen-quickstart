using Gen

##########
# step 1 #
##########

struct Point
    x::Float64
    y::Float64
end

function dist(a::Point, b::Point)
    dx = a.x - b.x
    dy = a.y - b.y
    sqrt(dx * dx + dy * dy)
end

function line_segments_intersect(a1::Point, a2::Point, b1::Point, b2::Point)
    if ccw(a1, a2, b1) * ccw(a1, a2, b2) > 0
        false
    elseif ccw(b1, b2, a1) * ccw(b1, b2, a2) > 0
        false
    else
        true
    end
end

ccw(a::Point, b::Point, c::Point) = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)

struct Obstacle
    vertices::Vector{Point}
end

function obstacle_intersects_line_segment(obstacle::Obstacle, a1::Point,  a2::Point)
    n = length(obstacle.vertices)
    for start_vertex_idx=1:n
        end_vertex_idx = start_vertex_idx % n + 1 # loop over to 1
        v1 = obstacle.vertices[start_vertex_idx]
        v2 = obstacle.vertices[end_vertex_idx]
        if line_segments_intersect(v1, v2, a1, a2)
            return true
        end
    end
    false
end

struct Scene
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    obstacles::Vector{Obstacle}
end

Scene(xmin, xmax, ymin, ymax) = Scene(xmin, xmax, ymin, ymax, Obstacle[])

add_obstacle!(scene, obstacle::Obstacle) = push!(scene.obstacles, obstacle)

function line_is_obstructed(scene::Scene, a1::Point, a2::Point)
    for obstacle in scene.obstacles
        if obstacle_intersects_line_segment(obstacle, a1, a2)
            return false
        end
    end
    true
end

function make_wall(vertical::Bool, start::Point, length::Float64, thickness::Float64)
    vertices = Vector{Point}(undef, 4)
    vertices[1] = start
    dx = vertical ? thickness : length
    dy = vertical ? length : thickness
    vertices[2] = Point(start.x + dx, start.y)
    vertices[3] = Point(start.x + dx, start.y + dy) 
    vertices[4] = Point(start.x, start.y + dy)
    Obstacle(vertices)
end 

function make_tree(center::Point, size::Float64)
    vertices = Vector{Point}(undef, 4)
    vertices[1] = Point(center.x - size/2, center.y - size/2)
    vertices[2] = Point(center.x + size/2, center.y - size/2)
    vertices[3] = Point(center.x + size/2, center.y + size/2)
    vertices[4] = Point(center.x - size/2, center.y + size/2)
    Obstacle(vertices)
end

##########
# step 2 #
##########

struct RRTNode
    conf::Point
    parent::Union{Nothing,RRTNode}
    control::Union{Nothing,Point}
    dist_from_start::Float64
end

struct RRT
    nodes::Vector{RRTNode}
end

function RRT(root_conf::Point)
    nodes = RRTNode[RRTNode(root_conf, nothing, nothing, 0.)]
    RRT(nodes)
end

add_node!(tree::RRT, node::RRTNode) = push!(tree.nodes, node)

root(tree::RRT) = tree.nodes[1]

function random_point_in_scene(scene::Scene)
    x = rand() * (scene.xmax - scene.xmin) + scene.xmin
    y = rand() * (scene.ymax - scene.ymin) + scene.ymin
    Point(x, y)
end

function nearest_node_on_tree(tree::RRT, conf::Point)
    nearest::RRTNode = root(tree)
    best_dist::Float64 = dist(nearest.conf, conf)
    for node in tree.nodes
        d = dist(node.conf, conf)
        if d < best_dist
            best_dist = d 
            nearest = node
        end
    end 
    nearest
end

function generate_rrt(scene::Scene, init::Point, iters::Int, dt::Float64)
    tree = RRT(init) # init is the root of tree
    for iter=1:iters
        rand_conf::Point = random_point_in_scene(scene)
        near_node::RRTNode = nearest_node_on_tree(tree, rand_conf)
        
        dist_to_target = dist(near_node.conf, rand_conf)
        diff = Point(rand_conf.x - near_node.conf.x, rand_conf.y - near_node.conf.y)
        distance_to_move = min(dt, dist_to_target)
        scale = distance_to_move / dist_to_target
        control = Point(scale * diff.x, scale * diff.y)

        # go in the direction of target_conf from start_conf 
        new_conf = Point(near_node.conf.x + control.x, near_node.conf.y + control.y)

        # test the obstacles
        failed = line_is_obstructed(scene, near_node.conf, new_conf)
        
        if !failed
            dist_from_start = near_node.dist_from_start + distance_to_move
            new_node = RRTNode(new_conf, near_node, control, dist_from_start)
            add_node!(tree, new_node)
        end
    end
    tree
end

function find_tree_route_to_dest(tree::RRT, dest::Point)
    best_node = tree.nodes[1]
    min_cost = Inf
    path_found = false
    for node in tree.nodes
        # check for line-of-site to the destination
        clear_path = !line_is_obstructed(scene, node.conf, dest)
        cost = node.dist_from_start + (clear_path ? dist(node.conf, dest) : Inf)
        if cost < min_cost
            path_found = true
            best_node = node
            min_cost = cost
        end
    end
    if path_found
        (best_node, min_cost)
    else
        (nothing, min_cost)
    end
end

struct Path
    points::Vector{Point}
end

function walk_tree_to_root(node::RRTNode, start::Point, dest::Point)
    points = Point[dest]
    while node.parent != nothing
        push!(points, node.conf)
        node = node.parent
    end
    push!(points, start)
    @assert points[end] == start # the start point
    @assert points[1] == dest
    Path(reverse(points))
end

function refine_path(scene::Scene, original::Path, iters::Int, std::Float64)
    # do stochastic optimization
    new_points = copy(original.points)
    num_interior_points = length(original.points) -2
    if num_interior_points == 0
        return original
    end
    for i=1:iters
        point_idx = 2 + (i % num_interior_points)
        @assert point_idx > 1 # not start
        @assert point_idx < length(original.points) # not dest
        prev_point = new_points[point_idx-1]
        point = new_points[point_idx]
        next_point = new_points[point_idx+1]
        adjusted = Point(point.x + randn() * std, point.y + randn() * std)
        cur_dist = dist(prev_point, point) + dist(point, next_point)
        ok_backward = !line_is_obstructed(scene, prev_point, adjusted)
        ok_forward = !line_is_obstructed(scene, adjusted, next_point)
        if ok_backward && ok_forward
            new_dist = dist(prev_point, adjusted) + dist(adjusted, next_point)
            if new_dist < cur_dist
                # accept the change
                new_points[point_idx] = adjusted
            end
        end
    end
    Path(new_points)
end

struct PlannerParams
    rrt_iters::Int
    rrt_dt::Float64 # the maximum proposal distance
    refine_iters::Int
    refine_std::Float64
end

function plan_path(start::Point, dest::Point, scene::Scene, params::PlannerParams)
    
    # Generate a rapidly exploring random tree
    tree = generate_rrt(scene, start, params.rrt_iters, params.rrt_dt)

    # Find a route from the root of the tree to a node on the tree that has a line-of-sight to the destination
    (maybe_node, min_cost) = find_tree_route_to_dest(tree, dest)
    
    if maybe_node == nothing
        
        # No route found
        nothing
    else
        
        # Route found
        node::RRTNode = something(maybe_node)
        path = walk_tree_to_root(node, start, dest)
        refine_path(scene, path, params.refine_iters, params.refine_std)
    end
end

function compute_distances_from_start(path::Path)
    distances_from_start = Vector{Float64}(undef, length(path.points))
    distances_from_start[1] = 0.0
    for i=2:length(path.points)
        distances_from_start[i] = distances_from_start[i-1] + dist(path.points[i-1], path.points[i])
    end
    return distances_from_start
end

function walk_path(path::Path, speed::Float64, times::Array{Float64,1})
    distances_from_start = compute_distances_from_start(path)
    locations = Vector{Point}(undef, length(times))
    locations[1] = path.points[1]
    for (time_idx, t) in enumerate(times)
        if t < 0.0
            error("times must be positive")
        end
        desired_distance = t * speed
        used_up_time = false
        # NOTE: can be improved (iterate through path points along with times)
        for i=2:length(path.points)
            prev = path.points[i-1]
            cur = path.points[i]
            dist_to_prev = dist(prev, cur)
            if distances_from_start[i] >= desired_distance
                # we overshot, the location is between i-1 and i
                overshoot = distances_from_start[i] - desired_distance
                @assert overshoot <= dist_to_prev
                past_prev = dist_to_prev - overshoot
                frac = past_prev / dist_to_prev
                locations[time_idx] = Point(prev.x * (1. - frac) + cur.x * frac,
                                     prev.y * (1. - frac) + cur.y * frac)
                used_up_time = true
                break
            end
        end
        if !used_up_time
            # sit at the goal indefinitely
            locations[time_idx] = path.points[end]
        end
    end
    locations
end

##########
# step 3 #
##########

@gen function model(scene::Scene, measurement_times::Vector{Float64})

    # sample the start point of the agent from the prior
    start_x = @addr(uniform(0, 1), :start_x)
    start_y = @addr(uniform(0, 1), :start_y)
    start = Point(start_x, start_y)

    # sample the destination point of the agent from the prior
    dest_x = @addr(uniform(0, 1), :dest_x)
    dest_y = @addr(uniform(0, 1), :dest_y)
    dest = Point(dest_x, dest_y)

    # plan a path that avoids obstacles in the scene
    maybe_path::Union{Nothing,Path} = plan_path(start, dest, scene, PlannerParams(300, 3.0, 2000, 1.))
    
    # sample the speed from the prior
    speed = @addr(uniform(0, 1), :speed)

    if maybe_path == nothing
        
        # path planning failed, assume the agent stays as the start location indefinitely
        locations = fill(start, length(measurement_times))
    else
        
        # path planning succeeded, move along the path at constant speed
        path = something(maybe_path)
        locations = walk_path(path, speed, measurement_times)
    end

    # generate noisy measurements
    noise = 0.1 * @addr(uniform(0, 1), :noise)
    for (i, point) in enumerate(locations)
        @addr(normal(point.x, noise), i => :x)
        @addr(normal(point.y, noise), i => :y)
    end

    (start, dest, speed, noise, maybe_path, locations)
end

const scene = Scene(0, 1, 0, 1)
add_obstacle!(scene, make_tree(Point(0.30, 0.20), 0.1))
add_obstacle!(scene, make_tree(Point(0.83, 0.80), 0.1))
add_obstacle!(scene, make_tree(Point(0.80, 0.40), 0.1))
horizontal = false
vertical = true
wall_thickness = 0.02
add_obstacle!(scene, make_wall(horizontal, Point(0.20, 0.40), 0.40, wall_thickness))
add_obstacle!(scene, make_wall(vertical, Point(0.60, 0.40), 0.40, wall_thickness))
add_obstacle!(scene, make_wall(horizontal, Point(0.60 - 0.15, 0.80), 0.15 + wall_thickness, wall_thickness))
add_obstacle!(scene, make_wall(horizontal, Point(0.20, 0.80), 0.15, wall_thickness))
add_obstacle!(scene, make_wall(vertical, Point(0.20, 0.40), 0.40, wall_thickness))



function trace_to_dict(trace)
    args = get_args(trace)
    (scene, measurement_times) = args

    retval = get_retval(trace)
    (start, dest, speed, noise, maybe_path, locations) = retval

    d = Dict()
    d["scene"] = scene
    if maybe_path != nothing
        d["path"] = maybe_path.points
    end

    # the polygons for the obstacles

    # points
    #start = retval.start
    #dest = retval.dest
    #locations = retval.locations
    d
end

using GenViz
xs = collect(1:10)
ys = -collect(1:10) + randn(10)
server = VizServer(8000)
sleep(1)
viz = Viz(server, joinpath(@__DIR__, "vue/dist"), [])
openInBrowser(viz)
sleep(5)

const times = collect(range(0, stop=1, length=20))

for i=1:9

    (trace, _) = initialize(model, (scene, times))
    putTrace!(viz, i, trace_to_dict(trace))
end
readline()
