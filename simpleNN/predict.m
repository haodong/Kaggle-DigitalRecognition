function p = predict(net, X)

%	unroll theta
m = size(X, 1);
n = length(net.layers)-1;
nFrom = 1;
h = X;
for i = 1:n
	nCol = net.layers(i)+1;
	nRow = net.layers(i+1);

	nTo = nFrom + nRow * nCol - 1;
	theta = reshape(net.theta(nFrom:nTo), nRow, nCol);
	h = sigmoid([ones(m, 1) h] * theta');
	nFrom = nTo + 1;
end

[dummy, p] = max(h, [], 2);

end
