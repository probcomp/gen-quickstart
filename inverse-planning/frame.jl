import Luxor
using  Luxor: Point

# A Frame defines a particular coordinate system
# for points within a rectangle. No assumptions are made 
# about whether increasing x (and y) cooridnates correspond to 
# moving left or right (and up or down).
struct Frame
    top_left     :: Point
    bottom_right :: Point

    function Frame(top_left, bottom_right)
        new(top_left, bottom_right)
    end

    function Frame(left_x, right_x, bottom_y, top_y)
        new(Point(left_x, top_y), Point(right_x, bottom_y))
    end
end

# A Frame representing Luxor's coordinate system for a scene with a certain 
# width and height.
function luxor_frame(width, height)
    Frame(Point(-width/2, -height/2), Point(width/2, height/2))
end

# Default size parameters in pixels
DEFAULT_IMAGE_HEIGHT = 600
DEFAULT_IMAGE_WIDTH  = 600


function frame_converter(source_frame::Frame, target_frame::Frame) 
    function convert_frame(pt::Point)
        target_x = (pt.x - source_frame.top_left.x)     / (source_frame.bottom_right.x - source_frame.top_left.x) * (target_frame.bottom_right.x - target_frame.top_left.x) + target_frame.top_left.x
        target_y = (pt.y - source_frame.bottom_right.y) / (source_frame.top_left.y - source_frame.bottom_right.y) * (target_frame.top_left.y - target_frame.bottom_right.y) + target_frame.bottom_right.y

        Point(target_x, target_y)
    end
end

function draw_grid(element_drawer, elements, num_columns, total_width, element_aspect_ratio; separators=nothing)
    width_per_element = floor(total_width / num_columns)
    height_per_element = width_per_element / element_aspect_ratio
    num_rows = Int(ceil(length(elements) / num_columns))
    total_height = height_per_element * num_rows

    logical_frame = Frame(Point(0, 0), Point(total_width, total_height))
    to_luxor = frame_converter(logical_frame, luxor_frame(total_width, total_height))

    for row in 1:num_rows
        for col in 1:num_columns
            element_number = num_columns * (row - 1) + col
            if element_number <= length(elements)
                element = elements[element_number]

                top_left_point     = to_luxor(Point(width_per_element * (col-1), height_per_element * (row-1)))
                bottom_right_point = to_luxor(Point(width_per_element * col,     height_per_element * row))

                target_frame = Frame(top_left_point, bottom_right_point)
                element_drawer(element, target_frame)
            end
        end
    end

    if !isnothing(separators)
        Luxor.setcolor(separators)

        for col in 1:num_columns-1
            Luxor.line(to_luxor(Point(width_per_element * col, 0)), to_luxor(Point(width_per_element * col, total_height)), :stroke)
        end

        for row in 1:num_rows-1
            Luxor.line(to_luxor(Point(0, height_per_element * row)), to_luxor(Point(total_width, height_per_element * row)), :stroke)
        end
    end
 end


function pad_frame(frame, percent=0.1)
    Frame(
        Point(frame.top_left.x + percent * (frame.bottom_right.x - frame.top_left.x),
              frame.top_left.y + percent * (frame.bottom_right.y - frame.top_left.y)),
        Point(frame.bottom_right.x + percent * (frame.top_left.x - frame.bottom_right.x),
              frame.bottom_right.y + percent * (frame.top_left.y - frame.bottom_right.y))
    )
end
