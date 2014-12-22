function [lambdaVec, errorTrain, errorVal] = validationCurve(initial_nn_params, layer_size, X, y, Xval, yval, lambdaVec)

	if ~exist('lambdaVec')
		lambdaVec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
	end
	n = length(lambdaVec)

	errorTrain = zeros(n, 1);
	errorVal = zeros(n, 1);

	for i = 1:n
		lambda = lambdaVec(i);
		nn_params = train(layer_size, X, y, lambda, initial_nn_params);
		errorTrain(i) = nnCostFunction(nn_params, layer_size, X, y, 0);
		errorVal(i) = nnCostFunction(nn_params, layer_size, Xval, yval, 0);
	end

end
