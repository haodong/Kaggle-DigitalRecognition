function test(net, X, y, Xval, yval, Xtest, ytest)

	pred = predict(net, X);
	cost(1) = nnCostFunction(net, X, y);
	accuracy(1) = mean(double(pred == y)) * 100;
	a = bsxfun(@eq, pred, unique(y)');
	b = bsxfun(@eq, y, unique(y)');
	valF1(1) = calF1(a, b);

	if exist('Xval') && exist('yval')
		pred = predict(net, Xval);
		cost(2) = nnCostFunction(net, Xval, yval);
		accuracy(2) = mean(double(pred == yval)) * 100;
		a = bsxfun(@eq, pred, unique(yval)');
		b = bsxfun(@eq, yval, unique(yval)');
		valF1(2) = calF1(a, b);
	end

	if exist('Xtest') && exist('ytest')
		pred = predict(net, Xtest);
		cost(3) = nnCostFunction(net, Xtest, ytest);
		accuracy(3) = mean(double(pred == ytest)) * 100;
		a = bsxfun(@eq, pred, unique(ytest)');
		b = bsxfun(@eq, ytest, unique(ytest)');
		valF1(3) = calF1(a, b);
	end

	fprintf('Training Set, Cross Validation Set, Test Set seperately...\n');
	fprintf('Error\t\tAccuracy\tF1\n');
	for i = 1:length(cost)
			fprintf(' %f\t%f\t%f\n', cost(i), accuracy(i), valF1(i));
	end
end