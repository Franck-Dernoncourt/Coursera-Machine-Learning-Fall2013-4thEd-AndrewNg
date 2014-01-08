function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];
% X = X(1, :);

% Initialize the hidden layers
num_hidden_layer = 1;
hidden_layer = {num_hidden_layer};
for hidden_layer_number = 1:num_hidden_layer
    hidden_layer{hidden_layer_number} = zeros();
end

% Initialize thetas
theta{1} = Theta1;
theta{2} = Theta2;

size(X)
size(Theta1)
size(Theta2)
hidden_layer{1} = [ones(m, 1) sigmoid(X*theta{1}')]; % Don't forget to add ones 
output = sigmoid(hidden_layer{1}*theta{2}');
size(output)
[~, p] = max(output, [], 2)


% =========================================================================


end
