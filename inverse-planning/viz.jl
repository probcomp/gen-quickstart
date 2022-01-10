include("frame.jl")

# Construct a frame based on a Scene object.
function Frame(scene::Scene)
    Frame(Point(scene.xmin, scene.ymax), Point(scene.xmax, scene.ymin))
end

# +
# Drawing scenes.
function draw_obstacle(obstacle, source_frame, target_frame)
    Luxor.setcolor("black")
    luxor_points = map(frame_converter(source_frame, target_frame), obstacle.vertices)
    Luxor.poly(luxor_points, :fill; close=true)
end

function draw_scene(scene, target_frame=luxor_frame(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT))
    for obstacle in scene.obstacles
        draw_obstacle(obstacle, Frame(scene), target_frame)
    end
end

function draw_path_line(points, convert::Function)
    for i in 2:length(points)
        pt1 = convert(points[i-1])
        pt2 = convert(points[i])
        Luxor.line(pt1, pt2, :stroke)
    end
end

function draw_start(scene::Scene, start::Point, target_frame=luxor_frame(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT); markersize=8, startopacity=1)
    convert = frame_converter(Frame(scene), target_frame)
    Luxor.setcolor(Luxor.sethue("darkturquoise")..., startopacity)
    Luxor.circle(convert(start), markersize, :stroke)
end

function draw_dest(scene::Scene, dest::Point, target_frame=luxor_frame(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT); markersize=8, destopacity=1)
    convert = frame_converter(Frame(scene), target_frame)
    Luxor.setcolor(Luxor.sethue("red2")..., destopacity)
    Luxor.star(convert(dest), markersize, 2, 1, 0, :fill) 
end

function draw_path(
        scene::Scene, start::Point, dest::Point,
        pathPoints::Vector, target_frame;
        markersize=8, pathopacity=1, startopacity=1, destopacity=1)
    convert = frame_converter(Frame(scene), target_frame)
 
    if !isempty(pathPoints)
        # Path line
        Luxor.setcolor(Luxor.sethue("orange")..., pathopacity)
        draw_path_line(pathPoints, convert)
    end       

    # Starting point
    draw_start(scene, start, target_frame; markersize=markersize, startopacity=startopacity)
    
    # Destination point
    draw_dest(scene, dest, target_frame; markersize=markersize, destopacity=destopacity)
end

function draw_measurements(scene::Scene, measurements::Vector{Point}, target_frame=luxor_frame(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT); markersize=8)
    convert = frame_converter(Frame(scene), target_frame)

    # Draw the measurements
    for measurement in measurements
        pos = convert(measurement)
        Luxor.setcolor("black")
        Luxor.setopacity(.8)
        Luxor.polycross(pos, markersize/1.5, 4, .15, pi/4, action=:fill)
        Luxor.setopacity(1)
    end
end

function draw_trace(d::Dict, target_frame=luxor_frame(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT); markersize=8, should_draw_path=true, should_draw_measurements=true)
    draw_path(d[:scene], d[:start], d[:dest], d[:path],
        target_frame; markersize=markersize, pathopacity=should_draw_path ? 1 : 0)
    should_draw_measurements && draw_measurements(d[:scene], d[:measurements], 
        target_frame; markersize=markersize)
    draw_scene(d[:scene], target_frame)
end

# +
# function draw_trace(trace::Dict, target_frame=luxor_frame(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT); draw_measurements=true, draw_path=true)
#     scene = trace[:scene]
#     start, dest = trace[:start], trace[:dest]
#     converter = frame_converter(Frame(scene), target_frame)

#     # Scene and path
#     draw_scene(scene, target_frame)
#     draw_path && draw_path_in_scene(scene, trace[:path], target_frame)

#     # Starting point
#     Luxor.setcolor("blue")
#     Luxor.circle(converter(start), 10, :fill)

#     # Destination point
#     Luxor.setcolor(0, 0, 0, 0.5)
#     Luxor.sethue("red")
#     Luxor.circle(converter(dest), 10, :fill)

#     # Draw the measurements
#     if draw_measurements
#         Luxor.setcolor("black")

#         for measurement in trace[:measurements]
#             pt = converter(measurement)
#             Luxor.star(pt, 10, 4, 0.25; action = :fill)
#         end
#     end
# end
# -

function trace_to_dict(trace)
    args = Gen.get_args(trace)
    (scene, _, num_ticks, _) = args
    choices = Gen.get_choices(trace)
    (planning_failed, maybe_path) = Gen.get_retval(trace)

    d = Dict()

    # scene (the obstacles)
    d[:scene] = scene

    # the points along the planned path
    if planning_failed
        d[:path] = []
    else
        d[:path] = maybe_path.points
    end

    # start and destination location
    d[:start] = Point(choices[:start_x], choices[:start_y])
    d[:dest]  = Point(choices[:dest_x],  choices[:dest_y])

    # the observed location of the agent over time
    measurements = Vector{Point}(undef, num_ticks)
    for i=1:num_ticks
        measurements[i] = Point(choices[:meas => (i, :x)], choices[:meas => (i, :y)])
    end
    d[:measurements] = measurements

    return d
end;

function draw_trace(trace::Trace, target_frame=luxor_frame(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT); draw_path=true, draw_measurements=false, markersize=8)
    draw_trace(trace_to_dict(trace), target_frame; should_draw_measurements=draw_measurements, should_draw_path=draw_path, markersize=markersize)
end

function visualize_grid(element_drawer, elements, num_columns, total_width=DEFAULT_IMAGE_WIDTH, element_aspect_ratio=1.0; separators=nothing)
    width_per_element = floor(total_width / num_columns)
    height_per_element = width_per_element / element_aspect_ratio
    num_rows = Int(ceil(length(elements) / num_columns))
    total_height = height_per_element * num_rows

    Luxor.@draw draw_grid(element_drawer, elements, num_columns, total_width, element_aspect_ratio; separators=separators) total_width total_height
end

function visualize(f; width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT)
    Luxor.@draw f() width height
end
