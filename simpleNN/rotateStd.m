function scalarMatrix = rotateStd(scalarMatrix, n_row);
	[m, n] = size(scalarMatrix);
	if ~exist('n_row')
		n_col = sqrt(n);
		n_row = n_col;
	else
		n_col = n / n_row;
	end
	for i = 1:m
		temp = reshape(scalarMatrix(i, :), n_row, n_col)';
		scalarMatrix(i, :) = temp(:);
	end
	%outputMatrix = zeros(numel(scalarMatrix),1);
	%Matrix = reshape(scalarMatrix, n_row, n_col);
	%[n_row n_col] = size(Matrix);
	%for i = 1:n_row
	%	to=(n_row+1-i)*n_col;
	%	from=to-n_col+1;
	%	outputMatrix(from:to)=Matrix(i,:);
	%end
	%outputMatrix = reshape(outputMatrix, n_col, n_row);
end