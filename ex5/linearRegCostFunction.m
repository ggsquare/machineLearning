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

hyp = X*theta;

J = (1/(2*m)) * sum((hyp - y) .^ 2);

%Now regularize
J = J + (lambda/(2*m)) * sum(theta(2:end) .^2);

grad(1) = (1/m)*sum((hyp-y)'*X(:,1));

%Tried fully vectorized implementation, this proved to be easier
%Regularized gradient, for j >=1
for j=2:size(grad)
	grad(j) = (1/m)*sum((hyp-y)'*X(:,j))+(lambda/m)*theta(j);
end;



% =========================================================================

grad = grad(:);

end
