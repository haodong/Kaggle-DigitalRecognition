function outputMatrix = rotaCo90byscalar(scalarMatrix, n_row, n_col);
	outputMatrix = zeros(numel(scalarMatrix),1);
	Matrix = reshape(scalarMatrix, n_row, n_col);
	%[n_row n_col] = size(Matrix);
	for i = 1:n_row
		to=(n_row+1-i)*n_col;
		from=to-n_col+1;
		outputMatrix(from:to)=Matrix(i,:);
	end
	%outputMatrix = reshape(outputMatrix, n_col, n_row);
end