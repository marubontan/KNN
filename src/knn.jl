using DataFrames, CSV, DataStructures
include("dist.jl")

type KNN
    x::DataFrames.DataFrame
    y::DataFrames.DataFrame
end

function predict(data::KNN, testData::DataFrames.DataFrame; k=5, method="euclidean")
    targetPointsNum = size(testData, 1)
    predictedLabels = Array{String}(targetPointsNum)
    for i in 1:targetPointsNum
        sourcePoint = Array(testData[i,:])
        trainPointsNum = size(data.x, 1)
        distances = Array{Float64}(trainPointsNum)
        for j in 1:trainPointsNum
            destPoint = Array(data.x[j,:])
            distance = calcDist(sourcePoint, destPoint; method=method)
            distances[j] = distance
        end
        sortedIndex = sortperm(distances)
        targetCandidates = Array(data.y)[sortedIndex[1:k]]
        predictedLabel = extractTop(targetCandidates)
        predictedLabels[i] = predictedLabel
    end
    return predictedLabels
end

function calcDist(sourcePoint::Array, destPoint::Array; method="euclidean")

    if length(sourcePoint) != length(destPoint)
        error("The lengths of two arrays are different.")
        return
    end

    if method == "euclidean"
        return euclidean(sourcePoint, destPoint)
    elseif method == "minkowski"
        return minkowski(sourcePoint, destPoint)
    end
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
