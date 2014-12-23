function net = train(net, X, y, nIte)
	% this function wholly calculate accuracy for specific lambda value in order to gain a better value.

%	unroll theta
n = length(net.layers);
%input_net.layers = size(X, 2);
%num_labels = length(unique(y));
if ~isfield(net, 'theta')
	net.theta = initializeWeight(net.layers);
end

costFunction = @(p) nnCostFunction(net, X, y, p);
if exist('nIte')
	options = optimset('MaxIter', nIte);
	net.theta = fmincg(costFunction, net.theta, options);
else
	net.theta = fmincg(costFunction, net.theta);
end

end