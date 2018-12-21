using Gen
using GenViz

@gen function my_model(xs::Vector{Float64})
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 10), :intercept)
    for (i, x) in enumerate(xs)
        @addr(normal(slope * x + intercept, 1), "y-$i")
    end
end

function trace_to_dict(trace)
    args = get_args(trace)
    num_data = length(args[1])
    assmt = get_assmt(trace)
    Dict("slope" => assmt[:slope], "intercept" => assmt[:intercept])
end

function my_inference_program(xs::Vector{Float64}, ys::Vector{Float64},
                              num_iters::Int, viz::Viz)
    constraints = DynamicAssignment()
    for (i, y) in enumerate(ys)
        constraints["y-$i"] = y
    end
    (trace, _) = initialize(my_model, (xs,), constraints)
    slope_selection = select(:slope)
    intercept_selection = select(:intercept)
    for iter=1:num_iters
        putTrace!(viz, 1, trace_to_dict(trace))
        (trace, _) = default_mh(trace, slope_selection)
        (trace, _) = default_mh(trace, intercept_selection)
    end
    assmt = get_assmt(trace)
    return (assmt[:slope], assmt[:intercept])
end


xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]

# start visualization server
server = VizServer(8000)
viz = Viz(server, joinpath(@__DIR__, "vue/dist"), Dict("xs" => xs, "ys" => ys, "num" => length(xs), "xlim" => [minimum(xs), maximum(xs)], "ylim" => [minimum(ys), maximum(ys)]))
println("Open http://localhost:8000/$(viz.id)/ in a browser to view visualization")

(slope, intercept) = my_inference_program(xs, ys, 100000000, viz)
println("slope: $slope, intercept: $slope")
