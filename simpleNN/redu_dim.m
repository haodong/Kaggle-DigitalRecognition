function Z = redu_dim(X, K, draw)
	m = size(X, 1);
	%[X_norm, mu, sigma] = featureNormalize(X);
	X_norm = X/255;
	X_norm = featureNormalize(X_norm);
	[U, S] = pca(X_norm);
	if exist('draw')
		displayData(U(:, 1:22:784)');
		
		fprintf('Program paused. Press enter to continue.\n');
		pause;
		close all;
	end

	if ~exist('K')
		K = 100;
	end
	Z = projectData(X_norm, U, K);
	if exist('draw')
		X_rec  = recoverData(Z, U, K);
		sel = randperm(size(X,1));
		sel = sel(1:100);

		% Display normalized data
		subplot(1, 2, 1);
		displayData(X_norm(sel,:));
		title('Original Images');
		axis square;

		% Display reconstructed data from only k eigenfaces
		subplot(1, 2, 2);
		displayData(X_rec(sel,:));
		title('Recovered Images');
		axis square;

		fprintf('Program paused. Press enter to continue.\n');
		pause;
		close all
	end
end