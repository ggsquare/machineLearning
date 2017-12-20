%Implementing the optimization
options = optimset('GradObj', 'on', 'MaxIter', '100');
initialTheta = zeros(2, 1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
disp("optTheta: "), disp(optTheta);
disp("functionVal: "), disp(functionVal);
disp("exitFlag: "), disp(exitFlag);

