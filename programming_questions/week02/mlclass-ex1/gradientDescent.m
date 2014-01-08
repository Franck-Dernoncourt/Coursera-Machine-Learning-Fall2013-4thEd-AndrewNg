function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
% num_iters=100;
J_history = zeros(num_iters, 1);
theta_history = zeros(num_iters, 2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    
    % http://stackoverflow.com/questions/10479353/gradient-descent-seems-to-fail
    
    % Non-vectorized:
%     theta_1 = theta(1) - alpha * (1/m) * sum((X*theta-y).*X(:,1))
%     theta_2 = theta(2) - alpha * (1/m) * sum((X*theta-y).*X(:,2))
    %     theta(1) = theta_1;
    %     theta(2) = theta_2; 
    
    % vectorized:
    theta = theta - (alpha .* (X * theta - y)'*X ./m)';



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    theta_history(iter, :) = theta';

end
% J_history
% figure
% plot(J_history)
% figure
% plot(1:num_iters, theta_history(:, 1),'b',1:num_iters,theta_history(:, 2),'r');
% figure
end
