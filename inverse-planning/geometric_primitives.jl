# First, we want a data type for a location in a two-dimensional plane:

import Base: @kwdef
using Luxor: Point

# We will need to measure the distance between two points, so we define a method for this:

function dist(a::Point, b::Point)
    dx = a.x - b.x
    dy = a.y - b.y
    sqrt(dx * dx + dy * dy)
end

# We will also need to test whether two line segments intersect. The method
# below tests whether the line segment between the points `a1` and `a2`
# intersects the segment between the points `b1` and `b2`. The implementation
# of this primitive is based upon https://algs4.cs.princeton.edu/91primitives/. 

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

# Next, we define a data type to represent obstacles in the scene. An obstacle
# will be represented by a polygon. We also define a method to test whether a
# given line defined by points `a1` and `a2` intersects the obstacle:

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

# Now, we define a data type to represent the two-dimensional scene. The scene
# spans a rectangle of on the two-dimensional x-y plane, and contains a list of
# obstacles. Each obstacle is a polygon defined by a list of vertex points. We
# also define a method to compute whether a given line is obstructed by any
# obstacles in the scene.

@kwdef struct Scene
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
            return true
        end
    end
    false
end

# Finally, we write some methods that allow us to concisely construct walls
# (line-shaped obstacles that are either vertically or horizontally oriented),
# and square-shaped obstacles (representing trees).

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
