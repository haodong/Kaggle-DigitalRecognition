function [lambdaVec, errorTrain, errorVal, initTheta] = validationCurve(net, X, y, Xval, yval, lambdaVec)

	if ~exist('lambdaVec')
		lambdaVec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
	end
	n = length(lambdaVec)

	errorTrain = zeros(n, 1);
	errorVal = zeros(n, 1);

	for i = 1:n
		netVal = net;
		netVal.lambda = lambdaVec(i);
		netVal = train(netVal, X, y);
		netVal.lambda = 0;
		errorTrain(i) = nnCostFunction(netVal, X, y);
		errorVal(i) = nnCostFunction(netVal, Xval, yval);
		if i==1
			initTheta = netVal.theta;
		elseif errorVal(i)<errorVal(i-1)
			initTheta = netVal.theta;
		end
			
	end

end
