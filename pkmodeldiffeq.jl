using DifferentialEquations,  Plots, Optim, LossFunctions, Random, Distributions, DiffEqParamEstim
path = dirname(@__FILE__)
cd(path)

#PK extravascular two compartment model

function pkf!(du,u,p,t)
 du[1] = - p[1] * u[1]
 du[2] =   p[1] * u[1] - p[2] * u[2] + p[3] * u[3] - p[4] * u[2]
 du[3] =   p[2] * u[2] - p[3] * u[3]
end
u0 = [3.0, 0.0, 0.0]
p = [0.2, 0.15, 0.05, 0.08]
#p[1] = 0.2 #Ka
#p[2] = 0.15 #K12
#p[3] = 0.05 #K21
#p[4] = 0.08 #Kel

tspan = (0.0, 100.0)
prob = ODEProblem(pkf!,u0,tspan, p)
sol = solve(prob)

plot(sol,vars=(1))
plot!(sol,vars=(2))
plt = plot!(sol,vars=(3))

png(plt,"./diff2comp.png")

#Model optimization

function pkf1!(du,u,p,t)
 du[1] = - p[1] * u[1]
 du[2] =   p[1] * u[1] - p[2] * u[2]
end

u0    = [3.0, 0.0]
p     = [0.2, 0.15]
tspan = (0.0, 50.0)
prob  = ODEProblem(pkf1!,u0,tspan, p)
sol   = solve(prob)

plot(sol,vars=(1))
plot!(sol,vars=(2))

x = float.(collect(0:1:50))
y = hcat(sol.(x)...)'[:,2] .* exp.(rand(Normal(), length(x)) ./ 8)

function loss_function(sol, x, y)
   tot_loss = 0.0
   if any((s.retcode != :Success for s in sol))
     tot_loss = Inf
   else
     #=
     for i = 1:length(x)
       tot_loss += value(LogitDistLoss(), sol(x[i])[2], y[i])
       #L2DistLoss
     end
     =#
     tot_loss = sum(value(HuberLoss(mean(y)), hcat(sol.(x)...)'[:,2], y))
   end
   tot_loss
end

#cost_function = build_loss_objective(prob, Tsit5(), L2Loss(x,y; data_weight=y ./ x))
cost_function = build_loss_objective(prob, Tsit5(), f ->  loss_function(f, x, y))

#result = optimize(cost_function, [0.4, 0.01], LBFGS())
#result = optimize(cost_function, [0.4, 0.01], Newton())
result = optimize(cost_function, [0.4, 0.01], NelderMead())
prob   = ODEProblem(pkf1!, u0, tspan, result.minimizer)
sol    = solve(prob)

plot!(sol,vars=(2))
plt = scatter!(x,y)

png(plt,"./diffOptim.png")

#Multiple dosing

dosetimes = [0.0, 24, 48, 72, 96]
condition(u,t,integrator) = t ∈ dosetimes
affect!(integrator) = integrator.u[1] += 10

cb = DiscreteCallback(condition,affect!)

function pkmf!(du,u,p,t)
 du[1] = - p[1] * u[1]
 du[2] =   p[1] * u[1] - p[2] * u[2]
end

u0 = [0.000, 0.000]
p = [0.3, 0.04]

tspan = (-0.001, 120.0)
prob = ODEProblem(pkmf!,u0,tspan, p)
sol = solve(prob, callback = cb, tstops = dosetimes)

plot(sol,vars=(1))
plt = plot!(sol,vars=(2))
png(plt,"./diffmult.png")
