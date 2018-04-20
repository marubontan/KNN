using Base.Test
using DataFrames
include("../src/knn.jl")
include("../src/dist.jl")

@testset "distance function" begin
    dataSourceNum = [1.0, 2.0]
    dataDestNum = [3.0, 4.0]
    @test euclidean(dataSourceNum, dataDestNum) == sqrt(8)
    @test minkowski(dataSourceNum, dataDestNum) == 4

    dataSourceStr = ["cat" "dog"]
    dataDestStr = ["human", "fish"]
    @test_throws MethodError euclidean(dataSourceStr, dataDestStr)
    @test_throws MethodError minkowski(dataSourceStr, dataDestStr)
end

@testset "KNN" begin
    @testset "getProb function" begin
        x = DataFrame(x = [1,2,3], y = [4,5,6])
        y = DataFrame(label = ['a','a','b'])
        sortedLabel = sort(unique(Array(y)))

        @test getProb(['a', 'a', 'a'], sortedLabel) == [1.0, 0.0]
    end
end
