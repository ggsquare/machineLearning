function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

%Calculate sigmoid value for all m examples
sig = sigmoid(X*theta);

for i=1:m
	if(sig(i) >= .5)
		p(i) = 1;
elseif(sig(i)< .5)
	p(i) = 0;
elseif(sig(i) < 0)
	disp("Error. Output of sigmoid can't be < 0.");
elseif(sig(i) > 1)
	disp("Error. Output of sigmoid can't be > 1.");
	end;
end;




% =========================================================================


end
