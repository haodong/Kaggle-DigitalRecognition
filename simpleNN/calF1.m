function F = calF1(pred, y)

	% True Positive
	TP = sum(pred == 1 & y == 1);
	% True Negative
	%TN = sum(pred == 0 & y == 0);
	% False Positive
	FP = sum(pred == 1 & y == 0);
	% False Negative
	FN = sum(pred == 0 & y == 1);

	precision = TP/(TP + FP);
	recall = TP/(TP + FN);

	if (TP==0)
		F = 0;
	else 
		F = 2 * precision * recall / (precision + recall);
	end

end