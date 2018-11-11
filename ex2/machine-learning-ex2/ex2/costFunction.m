function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J = (-1/m);
h = sigmoid(X*theta); %m*1
part1 = y.*log(h); %m*1
part2 = (1-y).*log(1-h); %m*1
parts = part1 + part2; %m*1
J = J.*sum(parts);

diff = h - y; %m*1
partial_update = X'*diff; %n*1
grad = (1/m)*partial_update; %n*1 






% =============================================================

end
