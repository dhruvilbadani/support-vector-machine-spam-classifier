function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

CSet = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaSet = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
minError = realmax();
CAtMin = 0.01;
sigmaAtMin = 0.01;

for i=1:8
    for j=1:8
        tryC = CSet(i,1);
        trySigma = sigmaSet(j,1);
        tryModel = svmTrain(X, y, tryC, @(x1, x2) gaussianKernel(x1, x2, trySigma)); 
        tryPrediction = svmPredict(tryModel, Xval);
        tryError = mean(double(tryPrediction ~= yval));
        if tryError < minError
            minError = tryError;
            CAtMin = tryC;
            sigmaAtMin = trySigma;
        end
    end
end

C = CAtMin;
sigma = sigmaAtMin;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
