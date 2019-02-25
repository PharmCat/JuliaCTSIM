using Distributions, Random, Distributed, Plots
rng = MersenneTwister(1234);                       #Закоментировать для стационарного сида
Random.seed!(rng)                                  #Закоментировать для стационарного сида
BIN = Binomial(30, 0.9)                            #Биномиальное распределение N, P для симуляции числа субъектов
Z   = Normal()                                     #Нормальное распределение m= 0, sd = 1, для симуляции индивидуальных данных
m1  = 5.0                                          #Среднее 1
sd1 = 1.5                                          #SD 1
m2  = 4.0                                          #Среднее 2
sd2 = 1.5                                          #SD 2
delta = -1.0                                       #Non-inf / superior margin

#функция вычисления ДИ
function meanDiffEV(m1::Real, s1::Real, n1::Real, m2::Real, s2::Real, n2::Real, alpha::Real)
    diff   = m1 - m2
    stddev = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    stderr = stddev * sqrt(1/n1 + 1/n2)
    d      = stderr * quantile(TDist(n1+n2-2), 1-alpha/2)
    return diff-d, diff+d, diff
end

#Power simulation 2 stage, 2 means
#функция симуляции
function  simP2st2Means(BIN, Z, m1, sd1, m2, sd2, delta; alpha1=0.024, alpha2=0.025, log10n=5)

    n1::Int = 0
    n2::Int = 0
    num = 0
    p1  = 0
    p2  = 0
    itn::Int = round(10^log10n)
    for i = 1:itn                                  #цикл из 10^itn симуляций

        n1  = rand(BIN)                            #Число субъектов группы 1
        n2  = rand(BIN)                            #Число субъектов группы 2
        d1 = Array{Float64, 1}(undef, n1)          #Выделение памяти под массивы d1, d2
        d2 = Array{Float64, 1}(undef, n2)

        for i1 = 1:n1                               #Заполняем массив группы 1 данными субъектов с указанными значениями m, sd
            d1[i1] =  rand(Z)*sd1+m1
        end
        for i1 = 1:n2                               #Тоже самое для группы 2 (значения m, sd другие)
            d2[i1] =  rand(Z)*sd2+m2
        end
        #cdf - плотность, TDist - распределения стьюдента, далее критическое значение t (можно разложить что бы было понятно) #Тест двухсторонний
        #if 2*cdf(TDist(length(d1)+length(d2)-2), (mean(d1) - mean(d2) + delta)/(sqrt(((length(d1)-1)*var(d1)+(length(d2)-1)*var(d2))/(length(d1)+length(d2)-2)*(1/length(d1)+1/length(d2))))) < 0.05
        #или с помощью вычисления ДИ d2-d1
        if  meanDiffEV(mean(d2), var(d2), length(d2), mean(d1), var(d1), length(d1), alpha1*2)[1] > delta
            num=num+1                              #если (p < 0.05) нижняя граница ДИ > margin то считаем исследование удачным +1 к числу успехов
            p1 = p1 + 1                            #количество успехов этапа 1
        else                                       #Если нет, делаем "добор"
            n1  = rand(BIN)                        #На второй этап можно изменить объем выборки, для простоты используется значение выборки 1го этапа
            n2  = rand(BIN)
            d21 = Array{Float64, 1}(undef, n1)     #Также выделям массивы для элементов добора
            d22 = Array{Float64, 1}(undef, n2)

            for i1 = 1:n1                           #Заполняем данными "добавку" для группы 1
                d21[i1] =  rand(Z)*sd1+m1
            end
            for i1 = 1:n2                           #Для группы 2
                d22[i1] =  rand(Z)*sd2+m2
            end
            append!(d1,d21)                        #Добавляем к предыдущим данным новые
            append!(d2,d22)

            #Вторая проверка
            #if 2*cdf(TDist(length(d1)+length(d2)-2), (mean(d1) - mean(d2) + delta)/(sqrt(((length(d1)-1)*var(d1)+(length(d2)-1)*var(d2))/(length(d1)+length(d2)-2)*(1/length(d1)+1/length(d2))))) < 0.0499*2
            if  meanDiffEV(mean(d2), var(d2), length(d2), mean(d1), var(d1), length(d1), alpha2*2)[1] > delta
                num=num+1                          #если со второго раза получилось +1
                p2 = p2 + 1                        #количество успехов этапа 2
            end
        end
    end
    return (num/itn*100), p1, p2                   #Количесво успешных делим на общее число, это будет мощность или альфа в зависимости от заданных начальных установок
end



#p1 = @async  simP2st2Means(BIN, Z, m1, sd1, m2, sd2, delta; log10n=5.0)
@time r = simP2st2Means(BIN, Z, m1, sd1, m2, sd2, delta; log10n=4)
print(r)
#println(fetch(p1))
redx = Array{Float64, 1}(undef,0)
redy = Array{Float64, 1}(undef,0)
bluex = Array{Float64, 1}(undef,0)
bluey = Array{Float64, 1}(undef,0)
for i=1:10000
    global red
    a1=round(rand()*0.05, digits=4)
    a2=rand()*0.05
    r = simP2st2Means(BIN, Z, m1, sd1, m2, sd2, delta; alpha1=a1, alpha2=a2, log10n=4)
    if r[1] < 5
        push!(bluex, a1)
        push!(bluey, a2)
    else
        push!(redx, a1)
        push!(redy, a2)
    end
end


plot(redx, redy,seriestype=:scatter,title="Alpha", marker = (:hexagon, 4, 0.6, :red, stroke(0)), legend=false)
plot!(bluex, bluey, seriestype=:scatter,  marker = (:circle, 4, 0.6, :blue, stroke(0)))
# png("plot1")
