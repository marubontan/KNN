using DataFrames, CSV, DataStructures


type KNN
    x::DataFrames.DataFrame
    y::DataFrames.DataFrame
end

function predict(data::KNN, testData::DataFrames.DataFrame, k=5)
    predictedLabels = []
    for i in 1:size(testData, 1)
        sourcePoint = Array(testData[i,:])
        distances = []
        for j in 1:size(data.x, 1)
            destPoint = Array(data.x[j,:])
            distance = calcDist(sourcePoint, destPoint)
            push!(distances, distance)
        end
        sortedIndex = sortperm(distances)
        targetCandidates = Array(data.y)[sortedIndex[1:k]]
        predictedLabel = extractTop(targetCandidates)
        push!(predictedLabels, predictedLabel)
    end
    return predictedLabels
end

function calcDist(sourcePoint, destPoint)
    sum = 0
    for i in 1:length(sourcePoint)
        sum += (destPoint[i] - sourcePoint[i]) ^ 2
    end
    dist = sqrt(sum)
    return dist
end

function extractTop(targetCandidates)
    targetFrequency = counter(targetCandidates)

    normValue = 0
    normKey = "hoge"

    for key in keys(targetFrequency)
        if targetFrequency[key] > normValue
            normKey = key
            normValue = targetFrequency[key]
        end
    end
    return normKey
end

function splitTrainTest(data, at = 0.7)
    n = nrow(data)
    ind = shuffle(1:n)
    train_ind = view(ind, 1:floor(Int, at*n))
    test_ind = view(ind, (floor(Int, at*n)+1):n)
    return data[train_ind,:], data[test_ind,:]
end

function main()
    df = readtable("iris.csv", header=true)

    trainData, testData = splitTrainTest(df)

    xTrain = trainData[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]
    yTrain = trainData[:, [:Species]]
    xTest = testData[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]
    yTest = testData[:, [:Species]]

    knn = KNN(xTrain, yTrain)
    predicted = predict(knn, xTest)

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
