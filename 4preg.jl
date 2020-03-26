using Optim, LossFunctions, Distributions, Plots

#Regression function
function log4reg(x, a, b, c, d)
    return a + (b - a)/(1 + (c/x)^d)
end
#Loss function for x, y data and v regression parameter
function loss_function(x, y, v)
    #L2DistLoss()
    #LogitDistLoss()
    #HuberLoss()
   sum(value(L2DistLoss(), map(val -> log4reg(val, v[1], v[2], v[3], v[4]), x), y))
end

# TEST DATA GENERATION
#x values
x  = collect(0:1:100)
#Model function with known parameters
testf(x) = log4reg(x, 2, 10, 50, 7)
#true y values
ty = testf.(x)
#y values for optimization
y  = (rand(Normal(), length(x)) .* 0.2) .+ ty
# END TEST DATA GENERATION

#generic function for data and v parameter for optimization
lf = v ->  loss_function(x, y, v)
o = optimize(lf, [0.0,0.0,0.0,0.0], [100.0,100.0,100.0,100.0], [1.0,1.0,1.0,1.0], Fminbox(GradientDescent()))
result = Optim.minimizer(o)

plot(x, ty)
plot!(x, y)

resf(x) = log4reg(x, result[1], result[2], result[3], result[4])
plot!(x, resf.(x))
