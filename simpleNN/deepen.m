function X = deepen(X, threshold)
	X(X>=threshold) = 255;
	X(X<threshold) = 0;
end