using Plots, Random, Distributions, ClinicalTrialUtilities

path = dirname(@__FILE__)
cd(path)

struct PKModel
    F::Real
    D::Real
    Vd::Real
    ka::Real
    ke::Real
end

function pkmodel(F, D, Vd, ka, ke, t)
    F*D*ka*(exp(-ke*t) - exp(-ka*t))/(Vd*(ka-ke))
end
function pkprofile(t, m)
    pkmodel(m.F, m.D, m.Vd, m.ka, m.ke, t)
end

NORM   = Normal()
EXPO   = Exponential(2.5)
EXPO2  = Exponential(5.0)

model    =  PKModel(1, 1, 1.0, 0.4, 0.2)
time     = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 36, 48]
#Single PK Profile model
timecont = collect(0:0.1:48)
conc     = map(x -> pkprofile(x, model), timecont)
png(plot(timecont, conc, legend = false),"./model.png")

p1 = plot()
p2 = plot()
p3 = plot()
p4 = plot()
for i = 1:10
    model    =  PKModel(1, 1, 1*exp(rand(NORM)*0.2), 0.4*exp(rand(NORM)*0.6), 0.2*exp(rand(NORM)*0.4))
    rtime    = ((time .* 60) .+ rand.(EXPO2)) ./ 60; rtime[1] = 0; rtime[end] = 48.0

    conc1    = map(x -> pkprofile(x, model), timecont)
    conc2    = map(x -> pkprofile(x, model), time)
    conc3    = map(x -> pkprofile(x, model), rtime)

    err      = exp.(rand(NORM, 15) .* 0.04)
    conc4    = conc3 .* err

    plot!(p1, timecont, conc1, legend = false)
    plot!(p2, time, conc2, legend = false)
    plot!(p3, rtime, conc3, legend = false)
    plot!(p4, rtime, conc4, legend = false)
end
png(p1,"./rmodels.png")
png(p2,"./rmodelstime.png")
png(p3,"./rmodelsrandtime.png")
png(p4,"./randpkprofile.png")

function runmodel()
    p = plot()

    AUC1 = Vector{Float64}(undef, 0)
    AUC2 = Vector{Float64}(undef, 0)

    bias = Vector{Float64}(undef, 0)
    for i = 1:10000
        rtime = ((time .* 60) .+ rand.(EXPO2)) ./ 60; rtime[1] = 0; rtime[end] = 48.0
        model =  PKModel(1, 1, 1*exp(rand(NORM)*0.2), 0.4*exp(rand(NORM)*0.6), 0.2*exp(rand(NORM)*0.4))
        err   = exp.(rand(NORM, 15) .* 0.035)
        conc  = map(x -> pkprofile(x, model), time) .* err
        rconc = map(x -> pkprofile(x, model), rtime) .* err

        pk1   = ClinicalTrialUtilities.pkimport(time, conc)
        pk2   = ClinicalTrialUtilities.pkimport(time, rconc)

        nca1  = ClinicalTrialUtilities.nca!(pk1)
        nca2  = ClinicalTrialUtilities.nca!(pk2)

        push!(AUC1, nca1[:AUClast])
        push!(AUC2, nca2[:AUClast])

        push!(bias, mean(AUC1) / mean(AUC2))
        #plot!(rtime, rconc)
    end
    println("Mean 1 = $(mean(AUC1)); Mean 2 = $(mean(AUC2))")
    p = plot(bias[18:end], legend = false)
    #p = histogram(AUC1)
    p
end

png(runmodel(),"./simmodel.png")
