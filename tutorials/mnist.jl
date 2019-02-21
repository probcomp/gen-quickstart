import Random

import MLDatasets
train_x, train_y = MLDatasets.MNIST.traindata()

mutable struct MNISTTrainDataLoader
    cur_id::Int
    order::Vector{Int}
end

MNISTTrainDataLoader() = MNISTTrainDataLoader(1, Random.shuffle(1:60000))

function next_batch(loader::MNISTTrainDataLoader, batch_size)
    x = zeros(Float64, batch_size, 784)
    y = Vector{Int}(undef, batch_size)
    for i=1:batch_size
        x[i, :] = reshape(train_x[:,:,loader.cur_id], (28*28))
        y[i] = train_y[loader.cur_id] + 1
        loader.cur_id += 1
        if loader.cur_id > 60000
            loader.cur_id = 1
        end
    end
    x, y
end

function load_mnist_test_set()
    test_x, test_y = MLDatasets.MNIST.testdata()
    N = length(test_y)
    x = zeros(Float64, N, 784)
    y = Vector{Int}(undef, N)
    for i=1:N
        x[i, :] = reshape(test_x[:,:,i], (28*28))
        y[i] = test_y[i]+1
    end
    x, y
end
