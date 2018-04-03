using Base.Test
using DataFrames
include("../src/knn.jl")

@testset "getProb function" begin
    x = DataFrame(x = [1,2,3], y = [4,5,6])
    y = DataFrame(label = ['a','a','b'])
    sortedLabel = sort(unique(Array(y)))

    @test getProb(['a', 'a', 'a'], sortedLabel) == [1.0, 0.0]
end
