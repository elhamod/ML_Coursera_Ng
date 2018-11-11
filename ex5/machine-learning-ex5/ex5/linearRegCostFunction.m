function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
J = (1/(2*m))*sum((h - y).^2);
diff = h - y; %m*1
partial_update = X'*diff; %n*1
grad = (1/m)*partial_update; %n*1 

factors = ones(length(theta),1); %n*1
factors(1) = 0;
factors = factors'; %1*n

J = J + (lambda./(2*m)).*(factors*(theta.^2)); %1*1
grad = grad + (lambda./(m)).*(factors'.*theta); %1*1








% =========================================================================

grad = grad(:);

end
