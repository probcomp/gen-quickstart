using Plots

function serialize_trace(trace)
    (xs,) = Gen.get_args(trace)
    Dict(:slope => trace[:slope],
         :intercept => trace[:intercept],
         :inlier_std => trace[:noise],
         :points => zip(xs, [trace[:data => i => :y] for i in 1:length(xs)]),
         :outliers => [trace[:data => i => :is_outlier] for i in 1:length(xs)])
end


function visualize_trace(trace::Trace; title="")
    trace = serialize_trace(trace)

    outliers = [pt for (pt, outlier) in zip(trace[:points], trace[:outliers]) if outlier]
    inliers =  [pt for (pt, outlier) in zip(trace[:points], trace[:outliers]) if !outlier]
    Plots.scatter(map(first, inliers), map(last, inliers), markercolor="blue", label=nothing, xlims=[-5, 5], ylims=[-20, 20], title=title) 
    Plots.scatter!(map(first, outliers), map(last, outliers), markercolor="red", label=nothing)

    inferred_line(x) = trace[:slope] * x + trace[:intercept]
    left_x = -5
    left_y  = inferred_line(left_x)
    right_x = 5
    right_y = inferred_line(right_x)
    Plots.plot!([left_x, right_x], [left_y, right_y], color="black", label=nothing)

    # Inlier noise
    inlier_std = trace[:inlier_std]
    noise_points = [(left_x, left_y + inlier_std),
                    (right_x, right_y + inlier_std),
                    (right_x, right_y - inlier_std),
                    (left_x, left_y - inlier_std)]
    Plots.plot!(Shape(map(first, noise_points), map(last, noise_points)), color="black", alpha=0.2, label=nothing)
    Plots.plot!(Shape([-5, 5, 5, -5], [10, 10, -10, -10]), color="black", label=nothing, alpha=0.08)
end
