using DataFrames, CSV, StatsModels, Random,  ForwardDiff, LinearAlgebra, Optim, BenchmarkTools

#THIS IS CASTOM SOLUTION FOR EDUCATION PURPOSE!!!

mixdata = """subject,repeat,effect,response
1,1,1,1
1,2,1,1.1
1,1,2,1.2
1,2,2,1.3
2,1,1,1.1
2,2,1,1
2,1,2,1.3
2,2,2,1.2
3,1,1,1.2
3,2,1,1.3
3,1,2,1.4
3,2,2,1.5
4,1,1,1.3
4,2,1,1.2
4,1,2,1.5
4,2,2,1.4
5,1,1,1.5
5,2,1,1.8
5,1,2,1.9
5,2,2,1.8
6,1,1,1.9
6,2,1,1.8
6,1,2,1.8
6,2,2,1.9"""
df = CSV.read(IOBuffer(mixdata)) |> DataFrame
#if isfile(filep) df = CSV.File(filep, delim=',') |> DataFrame end
categorical!(df, :subject);
categorical!(df, :repeat);
categorical!(df, :effect);
sort!(df,[:subject, :effect, :repeat])
"""
    X - fixed effect matrix
"""
X = ModelMatrix(ModelFrame(@formula(response ~ repeat + effect), df)).m
"""
    Z matrix of random effect
    for random effect we choose effect|subject
    and it is consructed with FullDummyCoding()
"""
Z = ModelMatrix(ModelFrame(@formula(response ~ 0+effect), df, contrasts = Dict(:effect => StatsModels.FullDummyCoding()))).m
"""
    Z and X matrix is equal for each subject,
    thats why in this castom example used truncated X and Z matrix.
    In general solution matrices can be unequal.
"""
Z = Z[1:4,:]
X = X[1:4,:]
"""
    Response vector y
"""
y = df[:, :response]
"""
    Construct qxq G matrix with known parameters,
    where q - length of random effect vector.
    For this case random vector is length 2, because
    factor effect have 2 levels.
    G matrix contructet as CSH: heterogeneous compound symmetry
    If form with σ₁, σ₂, σ₃, ... σₙ - variance for each level, and ρ - coeficient
    so in this case matrix G constructed like:
    σ₁²     σ₁σ₂ρ
    σ₁σ₂ρ   σ₂²
"""
function gmat(σ₁, σ₂, ρ)
    if ρ > 1.0 ρ = 1.0 end       #This privent optimiation algo to make matrix with negative variance
    if σ₁ < 0.0 σ₁ = 1.0e-6 end
    if σ₂ < 0.0 σ₂ = 1.0e-6 end
    cov = sqrt(σ₁ * σ₂) * ρ
    return [σ₁ cov; cov σ₂]
end
"""
    R matrix describes intra-subject variability
    R matrix can be equal for each subject,
    because it depends from Z matrix
    In this case we have equal variance
    for each observation within random factor.
"""
function rmat(σ₁, σ₂)
    if σ₁ < 0.0 σ₁ = 1.0e-6 end
    if σ₂ < 0.0 σ₂ = 1.0e-6 end
    return [σ₁ 0 0 0; 0 σ₁ 0 0; 0 0 σ₂ 0; 0 0 0 σ₂]
end
"""
    Construct variance-covariance matrix
"""
function cov(G, R, Z)
    V  = Z*G*Z' + R
end
"""
    return β - fixed factor coefficients
"""
function βcoef(y, X, iV)
    n = 6                     #because we have 6 subjects with equal X and V martices we make this cheating
    A = inv((X'*iV*X) .* n)   #in general case it should be Σ₁ⁿ (Xᵢ' Vᵢ⁻¹  Xᵢ)
    β = zeros(rank(X))
    for i = 1:n
        yi = y[((i-1)*4 + 1):((i-1)*4 + 4)]  #response for subject
        β = β .+ X'*iV*yi     #here we use X and V⁻¹ matrix equal for each subject, but in other case it could be subject dependet matrices
    end
    return A*β
end
"""
    REML function
    this function not includes X'X part because it is constant
    y, Z, X - is known
    θvec - unknown vector of variance components
"""
function reml(y, Z, X, θvec)
    n = 6                                #do the same as in  βcoef
    G = gmat(θvec[3], θvec[4], θvec[5])
    R = rmat(θvec[1], θvec[2])
    V  = Z*G*Z' + R
    iV = inv(V)                          #inverted V
    c = 0
    c  = (24-3)/2*log(2π)
    θ1 = 0
    θ2 = 0
    θ3 = 0
    θ2 = log(det((X'*iV*X) .* n))        #castom case: this trick is because Xᵢ' Vᵢ⁻¹  Xᵢ is equal for all subjects
    β  = βcoef(y, X, iV)
    for i = 1:n
        θ1 += log(det(V))
        r   = y[((i-1)*4 + 1):((i-1)*4 + 4)]-X*β
        θ3 += r'*iV*r
    end
    return -(θ1/2 + θ2/2 + θ3/2 + c)
end

"""
    Starting values, you can take any values you want,
    but in best case it should be near to expected
"""
θvec0 = [0.045833, 0.045833, 0.045833, 0.045833, 0.01]

"""
    We will optimize this function (-2 REML, as many software do this)
"""
remlf(x) = -2*reml(y, Z, X, x)

#we will find this:
#θvec  = [0.009650, 0.004798, 0.1045, 0.07275, 0.999999]

O = optimize(remlf, θvec0, method=Newton(),  g_tol= 1e-10, x_tol = 1e-10, store_trace = true, extended_trace = true, show_trace = false)

#Or we can use gradient free method:
#O = optimize(remlf, [1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6], [Inf, Inf, Inf, Inf, 1.0], θvec0, Fminbox(NelderMead()))

#=
Or we can use manual solution with Newton method:
    H  = ForwardDiff.hessian(remlf, θ)
    g  = ForwardDiff.gradient(remlf, θ)
    θ₂ = θ₁ - inv(H)*g
    or with Levenberg-Marquardt correction
=#

θfinal = Optim.minimizer(O)
reml2  = remlf(θfinal)
G  = gmat(θfinal[3], θfinal[4], θfinal[5])
R  = rmat(θfinal[1], θfinal[2])
V  = cov(G, R, Z)
β  = βcoef(y, X, inv(V))

"""
    Hessian matrix
    we can get with AD differentiation or from otimize if we use Newton() method
"""
H = ForwardDiff.hessian(remlf, θfinal)
H[:, 5] .= 0  #we need this because ρ not included in var-covar matrix
H[5, :] .= 0

"""
    Contrast vector for "effect" factor
"""
L = [0 0 1]

C = pinv((X'*inv(V)*X)) #C matrix

"""
    Funciol LCL' from θ
    needs for DF computation
"""
lclg(x) = (L*(inv((X'*inv(cov(gmat(x[3], x[4], x[5]), rmat(x[1], x[2]), Z))*X)))*L')[1]
#Gradient vector of LCL' with values of  θfinal
g2 = ForwardDiff.gradient(lclg, θfinal)
"""
    DF by Satterthwaite:
    2(L*C*L')²/(g'Ag)
    where:
    A - covariance matrix 2*pinv(H)
    g - gradient vector of LCL'
"""
v = 2*((L*C*L')[1])^2/(g2'*(2*pinv(H))*g2)

"""
    SE of factor, determined by L
"""
se = sqrt((L*inv((X'*inv(V)*X)*6)*L')[1])    #castom case, look at 6

"""
    F - statistics
"""
F = (L*β)[1]/(L*C*L')[1]

println("θ:           ", θfinal)
println("REML2:       ", reml2)
println("se (effect): ", se)
println("F statistics:", F)
println("DF:          ", v)
println("-----------------------------------")
println("G:")
println(G)
println("R:")
println(R)
println("V:")
println(V)
