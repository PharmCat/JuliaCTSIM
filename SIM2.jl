using Distributions, Random, Plots, DataFrames, GLM
rng = MersenneTwister(1234);                       #Сид
Random.seed!(rng)
function sim()
    x = Int[]
    y = Float64[]
    s = Int[]
    BERN = Bernoulli(0.5)
    for i = 1:250
        push!(s, rand(BERN))
        push!(x, i)
        push!(y, mean(s))
    end
    return x, y
end
x, y  = sim()
data = DataFrame()
data.X = x
data.Y = y
formula = @formula(Y ~ X)
wlinm = glm(formula,data, Normal(), IdentityLink(), wts=Float64.(data.X))
linm = glm(formula,data, Normal(), IdentityLink())
plot(x,y,seriestype=:scatter, ylims = (0,1),title="Proportion", marker = (:circle, 3, 0.6, :blue, stroke(0)),legend=false)
reg(x) = coeftable(linm).cols[1][1]+x*coeftable(linm).cols[1][2]
wreg(x) = coeftable(wlinm).cols[1][1]+x*coeftable(wlinm).cols[1][2]
plot!(reg)
plot!(wreg)
