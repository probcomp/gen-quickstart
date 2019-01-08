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
