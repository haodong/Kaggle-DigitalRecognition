function nn_params = train(layer_size, X, y, lambda, initial_nn_params, nIte)
	% this function wholly calculate accuracy for specific lambda value in order to gain a better value.

%	unroll theta
n = length(layer_size);
%input_layer_size = size(X, 2);
%num_labels = length(unique(y));
if ~exist('initial_nn_params')
	initial_nn_params = initializeWeight(layer_size);
end

costFunction = @(p) nnCostFunction(p, layer_size, X, y, lambda);
if exist('nIte')
	options = optimset('MaxIter', nIte);
	nn_params = fmincg(costFunction, initial_nn_params, options);
else
	nn_params = fmincg(costFunction, initial_nn_params);
end

end