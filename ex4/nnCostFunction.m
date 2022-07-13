function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Convert the labels to one hot encodings
y = sparse(1:numel(y), y,1);

% ========== Feedforward ==========
% Add ones to the X data matrix (for the bias unit in input)
X = [ones(m, 1) X];

% Hidden layer calculation
% Transpose Theta1 to allow matrix multiplication
% hidden = sigmoid( (5000 x 401) * (401 x 25) ) = (5000 x 25)
hidden = sigmoid(X * Theta1');
% Add ones to hidden (for the bias unit in hidden layer)
% hidden = (5000 x 26)
hidden = [ones(size(hidden,1), 1) hidden];

% Output layer calculation
hypothesis = sigmoid(hidden * Theta2');

% ========== Cost function ==========
% Calculate cost
% The inside summation is over all output nodes (i.e 1 to num_labels)
% The outside summation is over all training instances
cost = (1 / m) * sum( sum( (-y .* log(hypothesis)) - ((1 -y) .* log(1 - hypothesis)) ));

% Calculate regularization term
% Note: we ignore the bias (first column of theta)
Theta1_tmp = Theta1(:,2:end);
Theta2_tmp = Theta2(:,2:end);

% The inside summation is over a row of theta (i.e one nodes weights)
% The outside summation is over rows of theta (i.e all nodes in a layer)
reg = (lambda / (2 * m)) * (sum(sum( Theta1_tmp .^2, 2)) + sum(sum( Theta2_tmp .^2, 2)));

% Add to cost for reqularized cost
J = cost + reg;

% ========== Backpropagation ==========

% Calculate the difference between the output and actual labels
output_sigma = hypothesis - y;

% Calculate the difference between the next layer (output) and hidden layer
% Note: we need the inputs to the hidden layer (X * Theta1') not its activation (sigmoid(X * Theta1'))
% so we need to add the bias vector to allow matrix multiplication.
hidden_inputs = [ones(size((X * Theta1'), 1), 1) (X * Theta1')];
hidden_sigma = (output_sigma * Theta2) .* sigmoidGradient(hidden_inputs);
% Remove the bias vectors
hidden_sigma = hidden_sigma(:,2:end);

% Calculate gradients for hidden and output layers
hidden_delta = (1 / m) .* (hidden_sigma' * X);
output_delta = (1 / m) .* (output_sigma' * hidden);

% Calculate regularisation term
% Note: we ignore the bias (first column of theta) but need dimension to allow for matrix multiplication
% so set to zeros
Theta1_tmp = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_tmp = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_reg = (lambda / m) * Theta1_tmp;
Theta2_reg = (lambda / m) * Theta2_tmp;

% Add regularisation terms to gradients
Theta1_grad = hidden_delta + Theta1_reg;
Theta2_grad = output_delta + Theta2_reg;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
