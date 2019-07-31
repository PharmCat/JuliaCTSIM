using ForwardDiff, LinearAlgebra, DiffResults, Plots, BenchmarkTools

func(β, x) = β[1]*exp.(-β[2]*x)
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
y = [12.0, 7.0, 5.0, 3.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.2]
#β1 = [20,0.5]
β0 = [10,0.1]
f(x) = func(β0, x)

plot(1:10, f.(1:10))
plot!(x, y)

#= Gradient for x[1]
f(β) = func(β, x[1])
g = ForwardDiff.gradient(f, β)
=#

#= Equivalent below
function gvec(f, x, β)
    j = Array{Float64, length(β)}(undef, length(x), length(β))
    for i = 1:length(x)
        fx(β) = f(β, x[i])
        j[i,:] = ForwardDiff.gradient(fx, β)
    end
    return j
end
j = gvec(func, x, β0)
=#

fx(β) = func(β, x)
function newtongauss(fx, β0, y)
    res   = DiffResults.JacobianResult(y, β0)
    cfg   = ForwardDiff.JacobianConfig(fx, β0)
    cvg   = 1.0
#Start
    while sqrt(cvg) > 0.00001
        ForwardDiff.jacobian!(res, fx, β0, cfg);
        j     = DiffResults.jacobian(res)                     # Jacobian
#DiffResults.value(res) == fx(β0)
        e     = y-DiffResults.value(res)                      # yᵢ-f(x)
        sse   = sum(abs2, e)                                  # Σ(yᵢ-f(x))²

        #=QR
        qro   = qr(j)
        qy    = Array(qro.Q)' * e
        cvg   = sum(abs2, qy) / sse
        ldiv!(UpperTriangular(qro.R), qy)
        β0    = β0 .+ qy
        =#

        #Демиденко Е.З. Линейная и нелинейная регрессии М.: Финансы и статистика, 1981. стр. 245
        q     = j'*e                 #q(x)/2
        p     = inv(j'*j)*q                                # increment/decrement
        β0    = β0 .+ p
        cvg   = sum(abs2, p)
    end
    return β0
end
β = newtongauss(fx, β0, y)

#plotting
f(x) = func(β, x)
plot!(1:10, f(1:10))
#------------------

#NLreg
function decrement!(δ, res)
    negativeresid = res.value
    rss = sum(abs2, negativeresid)
    qrfac = qr!(res.derivs[1])                        #QR from Jacobian
    lmul!(qrfac.Q', negativeresid)
    copyto!(δ, 1, negativeresid, 1, length(δ))
    cvg = sum(abs2, δ) / rss
    ldiv!(UpperTriangular(qrfac.R), δ)
    rss, sqrt(cvg)
end

β0 = [10,0.1]
fx(β) = func(β, x)
# D Bates
function dbnlreg(fx, β0, y)
    δ     = similar(β0)
    res   = DiffResults.JacobianResult(y, β0)
    cfg   = ForwardDiff.JacobianConfig(fx, β0)
    cvg   = 1.0
    while cvg > 0.00001
        ForwardDiff.jacobian!(res, fx, β0, cfg).value .-= y
        rss, cvg = decrement!(δ, res)
        β0 = β0 .- δ
    end
    return β0
end
βdb = dbnlreg(fx, β0, y)



b1 = @benchmark β = newtongauss(fx, β0, y)
b2 = @benchmark βdb = dbnlreg(fx, β0, y)
