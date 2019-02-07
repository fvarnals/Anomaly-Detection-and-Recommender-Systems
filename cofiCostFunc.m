function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

%find only rated Movies
predictions = X*Theta';
errors = (predictions-Y);
error_factor = errors.*R;
Theta_reg = Theta(:,1:num_features);
X_reg = X(:,1:num_features);
J = (0.5*(sum(sum(error_factor.^2)))) + ((lambda/2) *sum(sum(Theta_reg.^2)))...
+ (lambda/2)*sum(sum(X_reg.^2)); %double sum needed for sum reason?
                                    %otherwise gives 1x3 matrix
                                    %remember it's 0.5 * everything in this
X_grad = (error_factor*Theta) + (lambda*X_reg);
Theta_grad = (error_factor'*X) + (lambda*Theta_reg);















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
