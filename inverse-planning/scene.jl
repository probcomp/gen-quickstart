import Base: @kwdef

@kwdef struct Scene
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    obstacles::Vector{Obstacle}
end

Scene(; xmin, xmax, ymin, ymax) = Scene(xmin, xmax, ymin, ymax, Obstacle[])

add_obstacle!(scene, obstacle::Obstacle) = push!(scene.obstacles, obstacle)

function line_is_obstructed(scene::Scene, a1::Point, a2::Point)
    for obstacle in scene.obstacles
        if obstacle_intersects_line_segment(obstacle, a1, a2)
            return true
        end
    end
    false
end

function make_line(vertical::Bool, start::Point, length::Float64, thickness::Float64)
    vertices = Vector{Point}(undef, 4)
    vertices[1] = start
    dx = vertical ? thickness : length
    dy = vertical ? length : thickness
    vertices[2] = Point(start.x + dx, start.y)
    vertices[3] = Point(start.x + dx, start.y + dy) 
    vertices[4] = Point(start.x, start.y + dy)
    Obstacle(vertices)
end 

function make_square(center::Point, size::Float64)
    vertices = Vector{Point}(undef, 4)
    vertices[1] = Point(center.x - size/2, center.y - size/2)
    vertices[2] = Point(center.x + size/2, center.y - size/2)
    vertices[3] = Point(center.x + size/2, center.y + size/2)
    vertices[4] = Point(center.x - size/2, center.y + size/2)
    Obstacle(vertices)
end
