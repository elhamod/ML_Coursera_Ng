function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

C_init = 0.01;
init_p = 1;
while (C_init <= 30)
    sigma_init = 0.01;
    while (sigma_init <= 30)
        model= svmTrain(X, y, C_init, @(x1, x2) gaussianKernel(x1, x2, sigma_init));   
        predictions = svmPredict(model, Xval);
        p = mean(double(predictions ~= yval));
        if ( p < init_p)
            init_p = p;
            C = C_init;
            sigma = sigma_init;
        end
         sigma_init = 3*sigma_init;
    end
     C_init = 3*C_init;
end





% =========================================================================

end
