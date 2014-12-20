function test(nn_params, layer_size, X, y)

	pred = predict(nn_params, layer_size, X);
	accuracy = mean(double(pred == y)) * 100;
	fprintf('\nThe Accuracy for this Set is: %f\n', accuracy);

	a = bsxfun(@eq, pred, unique(y)');
	b = bsxfun(@eq, y, unique(y)');
	valF1 = calF1(a, b);
	fprintf('\nThe F1 for this Set is: %f\n', valF1);

end