function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
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
%               following parts
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

X = [ones(m,1) X];

%Convert y from (1-10) to binary output 
y_i = eye(num_labels);
y = y_i(y, :);

%Gradient variable
bigDelta1 = 0;
bigDelta2 = 0;

%For each training ex
for i=1:m
	%Feedforward
	a_1 = X(i,:);
	z_2 = Theta1*a_1';
	a_2 = [1; sigmoid(z_2)];
	z_3 = Theta2*a_2;
	hyp = sigmoid(z_3);   %this is also a_3

	J = J + (-y(i,:)*log(hyp)-(1-y(i,:))*log(1-hyp));

	%LINE 85
	
	%Backprop
	delta_3 = hyp - y(i,:)';
	delta_2 = (Theta2(:,2:end)' * delta_3) .* sigmoidGradient(z_2);

	%Accumulate
	bigDelta1 = bigDelta1 + delta_2 * a_1;
	bigDelta2 = bigDelta2 + delta_3 * a_2';
end; 

%Calculate Cost
J = J/m;

%Regularize the cost function to avoid overfitting (code may be simple)
%Remember, don't regularize bias term
Theta1(:, 1) = []; %removes first column of matrix
Theta2(:, 1) = [];
Theta1Reg = sum(Theta1(:) .^ 2);
Theta2Reg = sum(Theta2(:) .^ 2);

Reg = lambda/(2*m) * (Theta1Reg + Theta2Reg);

J = J + Reg;

%unegularized gradient
Theta1 = [ones(rows(Theta1), 1) Theta1];
Theta2 = [ones(rows(Theta2), 1) Theta2];
Theta1_grad = (1/m) * bigDelta1;
Theta2_grad = (1/m) * bigDelta2;

%Overide all rows except j = 0 to regularize
Theta1_grad(2:end,:) = (1/m) * bigDelta1(2:end,:) + (lambda/m)*Theta1(2:end,:);
Theta2_grad(2:end,:) = (1/m) * bigDelta2(2:end,:) + (lambda/m)*Theta2(2:end,:);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
