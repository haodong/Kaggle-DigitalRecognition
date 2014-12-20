function lambda = seLambda(initial_nn_params, layer_size, X, y, Xval, yval, lambdaVec)

	if exist('lambdaVec')
		[lambdaVec, errorTrain, errorVal] = validationCurve(initial_nn_params, layer_size, X, y, Xval, yval, lambdaVec);
	else
		[lambdaVec, errorTrain, errorVal] = validationCurve(initial_nn_params, layer_size, X, y, Xval, yval);
	end

	plot(lambdaVec, errorTrain, lambdaVec, errorVal);
	legend('Train', 'Cross Validation');
	xlabel('lambda');
	ylabel('Error');

	fprintf('lambda\t\tTrain Error\tValidation Error\n');
	for i = 1:length(lambdaVec)
			fprintf(' %f\t%f\t%f\n', ...
				lambdaVec(i), errorTrain(i), errorVal(i));
	end

	%fprintf('Program paused. Press enter to continue.\n');
	%pause;

	[a, b]=min(errorVal);
	lambda = lambdaVec(b)

end