function p = predict(nn_params, layer_size, X)

%	unroll theta
m = size(X, 1);
n = length(layer_size)-1;
nFrom = 1;
h = X;
for i = 1:n
	nCol = layer_size(i)+1;
	nRow = layer_size(i+1);

	nTo = nFrom + nRow * nCol - 1;
	theta = reshape(nn_params(nFrom:nTo), nRow, nCol);
	h = sigmoid([ones(m, 1) h] * theta');
	nFrom = nTo + 1;
end

[dummy, p] = max(h, [], 2);

end
