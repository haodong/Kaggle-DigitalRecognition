function test(nn_params, layer_size, X, y, Xval, yval, Xtest, ytest, lambda)

	pred = predict(nn_params, layer_size, X);
	cost(1) = nnCostFunction(nn_params, layer_size, X, y, lambda);
	accuracy(1) = mean(double(pred == y)) * 100;
	a = bsxfun(@eq, pred, unique(y)');
	b = bsxfun(@eq, y, unique(y)');
	valF1(1) = calF1(a, b);

	pred = predict(nn_params, layer_size, Xval);
	cost(2) = nnCostFunction(nn_params, layer_size, Xval, yval, lambda);
	accuracy(2) = mean(double(pred == yval)) * 100;
	a = bsxfun(@eq, pred, unique(yval)');
	b = bsxfun(@eq, yval, unique(yval)');
	valF1(2) = calF1(a, b);

	pred = predict(nn_params, layer_size, Xtest);
	cost(3) = nnCostFunction(nn_params, layer_size, Xtest, ytest, lambda);
	accuracy(3) = mean(double(pred == ytest)) * 100;
	a = bsxfun(@eq, pred, unique(ytest)');
	b = bsxfun(@eq, ytest, unique(ytest)');
	valF1(3) = calF1(a, b);

	fprintf('Training Set, Cross Validation Set, Test Set seperately...\n');
	fprintf('Error\t\tAccuracy\tF1\n');
	for i = 1:3
			fprintf(' %f\t%f\t%f\n', cost(i), accuracy(i), valF1(i));
	end
end