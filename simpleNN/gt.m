fprintf('Training ...\n');
nn_params = train(layer_size, Z, y, lambda, nn_params, 1000);
test(nn_params, layer_size, Z, y, Zval, yval, Ztest, ytest, lambda);
fprintf('Training completed.\n');

fprintf('Writing predicted results ...\n');
pred = predict(nn_params, layer_size, Zpred);
csvwrite('/Volumes/RamDisk/results.csv', pred);
csvwrite('/Volumes/RamDisk/nn_params.csv', nn_params);
fprintf('Task completed.\n');
