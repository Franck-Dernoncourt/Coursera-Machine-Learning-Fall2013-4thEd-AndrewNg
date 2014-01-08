% https://code.google.com/p/matlab-doc/source/browse/trunk/recallAndPrecision.m?spec=svn25&r=25
% produces a vector for recall rates and a vector for precision rates from
% a confusion matrix
function [recall, precision] = recallAndPrecision(confusionMatrix)

    recall = zeros(length(confusionMatrix), 1);
    precision = zeros(length(confusionMatrix), 1);

    for i = 1:length(recall)
        recall(i) = recallValue(confusionMatrix, i);
        precision(i) = precisionValue(confusionMatrix, i);
    end
    
function [recallValue] = recallValue(confusionMatrix, index)
    truePositives = confusionMatrix(index, index);
    totalPositives = 0;
    
    for i = 1:length(confusionMatrix)
        totalPositives = totalPositives + confusionMatrix(index, i);
    end
    recallValue = truePositives/totalPositives;
    
function [precisionValue] = precisionValue(confusionMatrix, index)
    truePositives = confusionMatrix(index, index);
    falseNegatives = 0;
    
    for i = 1:length(confusionMatrix)
        if (i ~= index)
            falseNegatives = falseNegatives + confusionMatrix(i, index);
        end
    end
    
    precisionValue = truePositives/(truePositives + falseNegatives);