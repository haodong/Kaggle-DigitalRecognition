function initial_nn_params = initializeWeight(layer_size)
	n = length(layer_size);
	initial_nn_params = [];
	for i = 2:n
		nIn = layer_size(i-1);
		nOut = layer_size(i);
		weightValue = randInitializeWeights(nIn, nOut);
		initial_nn_params = [initial_nn_params; weightValue(:)];
	end
end