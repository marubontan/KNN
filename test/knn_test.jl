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
    @testset "calcDistancesBetweenSourceAndDestinations" begin
        sourceA = [1, 2]
        destsA = [3 4; 5 6]
        @test calcDistancesBetweenSourceAndDestinations(sourceA, destsA) == [sqrt(8), sqrt(32)]
    end
    df = readtable("../data/iris.csv", header=true)

    trainData, testData = splitTrainTest(df)

    xTrain = trainData[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]
    yTrain = trainData[:, [:Species]]
    xTest = testData[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]
    yTest = testData[:, [:Species]]

    knn = KNN(xTrain, yTrain)
    predicted = predict(knn, xTest; k=3)
    @test isa(predicted, Array)
    @test length(predicted) == size(xTest)[1]
end
