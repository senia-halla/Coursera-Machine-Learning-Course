function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.


% Initialize some useful values
m = length(y);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as thetha
z = sigmoid(X*theta);

a1 = -y
h1 = log(z);

a2 = -(1-y);
h2 = log(1-z);

J = ((a1' * h1) + (a2' * h2))/m ;

Diff = z -y ;
grad = (X' * Diff)/m;
% =============================================================

end
