using Optim, Plots

#Concentration
a = [0.0, 0.5, 1.0, 2.0, 7.0, 8.0, 4.0, 2.0, 1.0, 0.5]
#Time
t = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 3.0, 5.0, 7.0, 9.0]
#Time for plot
time =  collect(0.01:0.1:10.1)
#Dose
dose = 100

#Base pk model formula
function pk(t, d, v, ka, ke)
    return d / v * (ka / (ka-ke)) * (exp(-t*ke)-exp(-t*ka))
end
#Sum of squares Σ(yᵢ-ȳ)²
f(x) = sum((a .- pk.(t, dose, x[1], x[2], x[3])) .^ 2)

#Optimization f(x)
o = optimize(f, [1.0, 0.001, 0.0001], method = NelderMead())
#Result
r = Optim.minimizer(o)
#Plotting
plot(t, a)
plot!(time, pk.(time, dose, r[1], r[2], r[3]))
