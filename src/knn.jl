using DataFrames, CSV, DataStructures
include("dist.jl")

type KNN
    x::Array
    y::Array
end

function predict(data::KNN, testData::DataFrames.DataFrame; k=5, method="euclidean", prob=false)
    targetPointsNum = size(testData, 1)
    if prob == true
        sortedLabel = sort(unique(Array(data.y)))
    end
    predictedLabels = []
    trainPointsNum = size(data.x, 1)
    for i in 1:targetPointsNum
        sourcePoint = Array(testData[i,:])
        distances = calcDistancesBetweenSourceAndDestinations(sourcePoint, data.x)
        sortedIndex = sortperm(distances)
        targetCandidates = Array(data.y)[sortedIndex[1:k]]
        predictedLabel = prob ? getProb(targetCandidates, sortedLabel) : extractTop(targetCandidates)
        push!(predictedLabels, predictedLabel)
    end
    return predictedLabels
end

function calcDistancesBetweenSourceAndDestinations(source::Array, destinations::Array)
    return [calcDist(source, destinations[i, :]) for i in 1:size(destinations)[1]]
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

function getProb(targetCandidates, sortedLabel)
    targetFrequency = counter(targetCandidates)
    probDictionary = Dict()
    for label in sortedLabel
        if label in keys(targetFrequency)
            probDictionary[label] = targetFrequency[label]
        else
            probDictionary[label] = 0.0
        end
    end
    sumArray = [probDictionary[key] for key in sortedLabel]
    probArray = sumArray / sum(sumArray)
    return probArray
end

function extractTop(targetCandidates)
    targetFrequency = counter(targetCandidates)

    normValue = zero(Int)
    normKey = ""

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
