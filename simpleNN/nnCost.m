function [J grad] = nnCost(nn_params, input_layer_size, hidden_Layer_size1, hidden_Layer_size2, num_labels, X, y, lambda)

% Fold Theta values with dim
n1 = hidden_Layer_size1 * (input_layer_size + 1);				%	idem
Theta1 = reshape(nn_params(1:n1), hidden_Layer_size1, (input_layer_size + 1));
n2 = hidden_Layer_size2 * (hidden_Layer_size1 + 1);		%	idem
Theta2 = reshape(nn_params((1 + n1):(n1 + n2)), hidden_Layer_size2, (hidden_Layer_size1 + 1));
%	n3 = num_labels * (hidden_Layer_size2 + 1)
Theta3 = reshape(nn_params((1 + n1 + n2):end), num_labels, (hidden_Layer_size2 + 1));

%	Setup some useful variables
m = size(X, 1);

%	Forward Propagation
	% This is for input layer
a1 = [ones(m, 1) X];			%	m * (input_layer_size + 1), n=input_layer_size
	% This is for hidden layer 1
z2 = a1 * Theta1';				%	m * hidden_Layer_size1
a2 = [ones(m, 1) sigmoid(z2)];	%	m * (hidden_Layer_size1 + 1)
	% This is for hidden layer 2
z3 = a2 * Theta2';				%	m * hidden_Layer_size2
a3 = [ones(m, 1) sigmoid(z3)];	%	m * (hidden_Layer_size2 + 1)
	% This is for output layer
z4 = a3 * Theta3';				%	m * num_labels
a4 = sigmoid(z4);				%	idem

%	Cost Function
ty = bsxfun(@eq, y, 1:num_labels);				%	idem
j1 = -ty .* log(a4) - (1 - ty) .* log(1 - a4);	%	idem
	%	Below two are scalar variables
j2 = sumsq(Theta1(:,2:end)(:)) + sumsq(Theta2(:,2:end)(:)) + sumsq(Theta3(:, 2:end)(:));
J = (sum(j1(:)) + j2 * lambda / 2) / m;

%	Backpropagation
d4 = a4 - ty;												%	m * num_labels
d3 = (d4 * Theta3(:, 2:end)) .* sigmoidGradient(z3);			%	m * hidden_Layer_size2
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);			%	m * hidden_Layer_size1
Delta3 = d4' * a3;											%	num_labels * (hidden_Layer_size2 + 1)
Delta2 = d3' * a2;											%	hidden_Layer_size2 * (hidden_Layer_size1 + 1)
Delta1 = d2' * a1;											%	hidden_Layer_size1 * (input_layer_size + 1)

%	Gradient
tempTheta1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];		%	hidden_Layer_size1 * (input_layer_size + 1)
tempTheta2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];		%	hidden_Layer_size2 * (hidden_Layer_size1 + 1)
tempTheta3 = [zeros(size(Theta3, 1), 1) Theta3(:, 2:end)];		%	num_labels * (hidden_Layer_size2 + 1)
Theta1_grad = (Delta1 + lambda * tempTheta1) / m;				%	hidden_Layer_size1 * (input_layer_size + 1)
Theta2_grad = (Delta2 + lambda * tempTheta2) / m;				%	hidden_Layer_size2 * (hidden_Layer_size1 + 1)
Theta3_grad = (Delta3 + lambda * tempTheta3) / m;				%	num_labels * (hidden_Layer_size2 + 1)

grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];		%	(n1 + n2 + n3) * 1


end
