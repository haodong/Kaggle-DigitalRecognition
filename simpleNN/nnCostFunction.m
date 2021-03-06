function [J grad] = nnCostFunction(net, X, y, nn_params)

% Setup some useful variables
m = size(X, 1);

%	unroll theta
n = length(net.layers);
nFrom = 1;
for i = 1:n-1
	varName = ['Theta', int2str(i)];

	nCol = net.layers(i)+1;
	nRow = net.layers(i+1);

	nTo = nFrom + nRow * nCol - 1;
	if ~exist('nn_params')
		varValue = reshape(net.theta(nFrom:nTo), nRow, nCol);
	else
		varValue = reshape(nn_params(nFrom:nTo), nRow, nCol);
	end
	eval([varName, ' = varValue;']);
	nFrom = nTo + 1;
end

%	forward propagation
aPar1 = [ones(m, 1) X];
for i = 2:n
	varA = ['aPar', int2str(i)];
	varZ = ['zPar', int2str(i)];
	eval([varZ, ' = aPar', int2str(i-1), ' * Theta', int2str(i-1), ''';']);
	if i == n
		eval([varA, ' = sigmoid(zPar', int2str(i), ');']);
	else
		eval([varA, ' = [ones(m, 1) sigmoid(zPar', int2str(i), ')];']);
	end	
end

%	cost
yTemp = bsxfun(@eq, y, 1:net.layers(end));
jBase = -yTemp .* eval(['log(aPar', int2str(n), ')']) - (1 - yTemp) .* eval(['log(1 - aPar', int2str(n), ')']);
jRegu = 0;
for i = 1:n-1
	eval(['jTemp = Theta', int2str(i), '(:, 2:end);']);
	jTemp = jTemp(:);
	jTemp = jTemp' * jTemp;
	jRegu = jRegu + jTemp;
end
J = (sum(jBase(:)) + jRegu * net.lambda / 2) / m;

%	backpropagation
eval(['delta', int2str(n), '= aPar', int2str(n), ' - yTemp;']);
for i = n-1:-1:1
	if i>1
		eval(['delta', int2str(i), ' = (delta', int2str(i+1), '* Theta', int2str(i), '(:, 2:end)) .* sigmoidGradient(zPar', int2str(i), ');'])
	end
	eval(['Delta', int2str(i), ' = delta', int2str(i+1), ''' * aPar', int2str(i), ';']);
end

%	gradient
grad = [];
for i = 1:n-1
	eval(['thetaTemp', int2str(i), ' = [zeros(size(Theta', int2str(i), ', 1), 1) Theta', int2str(i), '(:, 2:end)];']);
	eval(['Theta', int2str(i), '_grad = (Delta', int2str(i), ' + net.lambda * thetaTemp', int2str(i), ') / m;']);
	eval(['grad = [grad; Theta', int2str(i), '_grad(:)];']);
end


end
