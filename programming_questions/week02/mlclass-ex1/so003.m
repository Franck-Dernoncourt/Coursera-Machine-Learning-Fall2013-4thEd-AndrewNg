
Y = rand(3,1)
X = ones(length(Y),1)
Y*X'
reshape((Y*X')', [], 1)