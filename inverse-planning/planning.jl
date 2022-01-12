# First, we define the data types for the RRT. The tree is a list of nodes,
# each of which stores the point (`conf`), a reference to the parent node, the
# vector from the parent to the node (`control`), and the total distance from
# the start node to this node, following the edges of the tree
# (`dist_from_start`).

struct RRTNode
    conf::Point
    parent::Union{Nothing,RRTNode}
    control::Union{Nothing,Point}
    dist_from_start::Float64
end

# The RRT is a list of nodes, where the first node is the 'root' node:

struct RRT
    nodes::Vector{RRTNode}
end

function RRT(root_conf::Point)
    nodes = RRTNode[RRTNode(root_conf, nothing, nothing, 0.)]
    RRT(nodes)
end

function get_edges(tree::RRT)
    edges = Tuple{Point,Point}[]
        for node in tree.nodes
            if node.parent != nothing
                push!(edges, (node.parent.conf, node.conf))
            end
        end
    edges
end

add_node!(tree::RRT, node::RRTNode) = push!(tree.nodes, node)

root(tree::RRT) = tree.nodes[1]

# One of the key operations used in the RRT algorithm is generating a random
# point in the scene:

function random_point_in_scene(scene::Scene)
    x = rand() * (scene.xmax - scene.xmin) + scene.xmin
    y = rand() * (scene.ymax - scene.ymin) + scene.ymin
    Point(x, y)
end

# Another key operation is finding the point on the tree that is closest to
# some other point:

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

# Finally, we implement the RRT algorithm, which generates a RRT. It operates
# by randomly picking points from the scene and trying to connect them to the
# tree with a non-obstructed line-of-site.

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

struct Path
    points::Vector{Point}
end

function get_path_to_dest(tree::RRT, destination::Point)
    
    # find a node in the tree with a clear line-of-sight to the destination
    best_node = tree.nodes[1]
    min_cost = Inf
    path_found = false
    for node in tree.nodes
        clear_path = !line_is_obstructed(scene, node.conf, destination)
        cost = node.dist_from_start + (clear_path ? dist(node.conf, destination) : Inf)
        if cost < min_cost
            path_found = true
            best_node = node
            min_cost = cost
        end
    end
    
    if path_found
        
        # walk from the best node to the root of the tree to construct the path
        points = Point[destination]
        node = best_node
        while node.parent != nothing
            push!(points, node.conf)
            node = node.parent
        end
        push!(points, root(tree).conf)
        Path(reverse(points))
    else
        
        # return nothing if no path was found
        nothing
    end
end

function get_edges(path::Path)
    edges = Tuple{Point,Point}[]
    for i=1:length(path.points)-1
        push!(edges, (path.points[i], path.points[i+1]))
    end
    edges
end

function refine_path(scene::Scene, original::Path, iters::Int, std::Float64)
    new_points = copy(original.points)
    num_interior_points = length(original.points) -2
    if num_interior_points == 0
        return original
    end
    for i=1:iters
        
        # propose an adjustment to one of the interior points on the path (not the first or last point)
        point_idx = 2 + (i % num_interior_points)
        prev_point = new_points[point_idx-1]
        point = new_points[point_idx]
        next_point = new_points[point_idx+1]
        adjusted_point = Point(point.x + randn() * std, point.y + randn() * std)
        
        # check if the new path is obstructed by obstacles
        ok_backward = !line_is_obstructed(scene, prev_point, adjusted_point)
        ok_forward = !line_is_obstructed(scene, adjusted_point, next_point)
        
        # accept the adjustment if it is not obstructed by obstacles and it reduces the length of the path
        if ok_backward && ok_forward
            new_dist = dist(prev_point, adjusted_point) + dist(adjusted_point, next_point)
            cur_dist = dist(prev_point, point) + dist(point, next_point)
            if new_dist < cur_dist
                new_points[point_idx] = adjusted_point
            end
        end
    end
    Path(new_points)
end

@kwdef struct PlannerParams
    rrt_iters::Int
    rrt_dt::Float64
    refine_iters::Int
    refine_std::Float64
end

function plan_path(start::Point, dest::Point, scene::Scene, params::PlannerParams)
    
    # Generate a rapidly exploring random tree
    tree = generate_rrt(scene, start, params.rrt_iters, params.rrt_dt)

    # Find a route from the root of the tree to a node on the tree that has a line-of-sight to the destination
    maybe_path = get_path_to_dest(tree, dest)
    
    if maybe_path == nothing
        
        # No route found
        return nothing
    else
        
        # Route found
        path = something(maybe_path)
        refined_path = refine_path(scene, maybe_path, params.refine_iters, params.refine_std)
        return refined_path
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

function walk_path(path::Path, speed::Float64, dt::Float64, num_ticks::Int)
    distances_from_start = compute_distances_from_start(path)
    locations = Vector{Point}(undef, num_ticks)
    locations[1] = path.points[1]
    t = 0.
    for time_idx=1:num_ticks
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
        t += dt
    end
    locations
end
