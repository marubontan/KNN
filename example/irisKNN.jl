using DataFrames, CSV, DataStructures
include("../src/knn.jl")

function main()
    df = readtable("../data/iris.csv", header=true)

    trainData, testData = splitTrainTest(df)

    xTrain = trainData[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]
    yTrain = trainData[:, [:Species]]
    xTest = testData[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]
    yTest = testData[:, [:Species]]

    knn = KNN(xTrain, yTrain)
    predicted = predict(knn, xTest; method="minkowski", k=10)

    accurate = 0
    yTestArray = Array(yTest)
    for i in 1:length(predicted)
        if yTestArray[i] == predicted[i]
            accurate += 1
        end
    end
    println(accurate/length(predicted))
end

main()
