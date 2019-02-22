using Distributions, Random, Distributed
rng = MersenneTwister(1234);
Random.seed!(rng)
BIN = Binomial(30, 0.9) #Биномиальное распределение N, P для симуляции числа субъектов
Z   = Normal() #Нормальное распределение m= 0, sd = 1, для симуляции индивидуальных данных
m1  = 5.0
sd1 = 1.5
m2  = 4.0
sd2 = 1.5
delta = -1.0 #Non-inf / superior margin


function meanDiffEV(m1::Real, s1::Real, n1::Real, m2::Real, s2::Real, n2::Real, alpha::Real)
    diff   = m1 - m2
    stddev = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    stderr = stddev * sqrt(1/n1 + 1/n2)
    d      = stderr * quantile(TDist(n1+n2-2), 1-alpha/2)
    return diff-d, diff+d, diff

end


#Power simulation 2 stage, 2 means
function  simP2st2Means(BIN, Z, m1, sd1, m2, sd2, delta; log10n=5) #функция симуляции, обертка, для оптимального кода


    n1::Int = 0
    n2::Int = 0
    num = 0
    p1  = 0
    p2  = 0
    itn = 10^log10n
    for i = 1:itn   #цикл из 100000 симуляций

        n1  = rand(BIN)   #Число субъектов группы 1
        n2  = rand(BIN)   #Число субъектов группы 2
        d1 = Array{Float64, 1}(undef, n1)   #Выделение памяти под массив
        d2 = Array{Float64, 1}(undef, n2)

        for i = 1:n1                  #Заполняем массив группы 1 данными субъектов с указанными значениями m, sd
            d1[i] =  rand(Z)*sd1+m1
        end
        for i = 1:n2                  #Тоже самое для группы 2 (значения m, sd другие)
            d2[i] =  rand(Z)*sd2+m2
        end
        #cdf - плотность, TDist - распределения стьюдента, далее критическое значение t (можно разложить что бы понятней было, на так меньше аллокаций)
        #Тест двухсторонний
        #if 2*cdf(TDist(length(d1)+length(d2)-2), (mean(d1) - mean(d2) + delta)/(sqrt(((length(d1)-1)*var(d1)+(length(d2)-1)*var(d2))/(length(d1)+length(d2)-2)*(1/length(d1)+1/length(d2))))) < 0.05
        #или
        if  meanDiffEV(mean(d2), var(d2), length(d2), mean(d1), var(d1), length(d1), 0.040*2)[1] > delta
            num=num+1    #если p < 0.05 то считаем исследование удачным +1
            p1 = p1 + 1
        else #if n1 < 0  #Если нет, делаем "добор"
            n1  = rand(BIN)  #на второй этап можно изменить объем выборки, для простоты используется тоже значение что и для этапа 1
            n2  = rand(BIN)
            d21 = Array{Float64, 1}(undef, n1) #Также выделям масси
            d22 = Array{Float64, 1}(undef, n2)

            for i = 1:n1                    #Заполняем данными добавку для группы 1
                d21[i] =  rand(Z)*sd1+m1
            end
            for i = 1:n2                    #Для группы 2
                d22[i] =  rand(Z)*sd2+m2
            end
            append!(d1,d21)    #добавляем к предыдущим данным новые
            append!(d2,d22)

            #Вторая проверка
            #if 2*cdf(TDist(length(d1)+length(d2)-2), (mean(d1) - mean(d2) + delta)/(sqrt(((length(d1)-1)*var(d1)+(length(d2)-1)*var(d2))/(length(d1)+length(d2)-2)*(1/length(d1)+1/length(d2))))) < 0.0499*2
            if  meanDiffEV(mean(d2), var(d2), length(d2), mean(d1), var(d1), length(d1), 0.02*2)[1] > delta
                num=num+1
                p2 = p2 + 1
            end #если со второго раза получилось +1
        end
        d1 = nothing
        d2 = nothing
        d21 = nothing
        d22 = nothing
    end
    return (num/itn*100), p1, p2 #Количесво успешных делим на общее число, это будет мощность если заданы данные обладающие различием или альфа если данные заданы как удовлетворяющие 0 гипотезе
end



#p1 = @async  a(BIN, Z, m1, sd1, m2, sd2, delta)
#p2 = @async  a(BIN, Z, m1, sd1, m2, sd2)
#p3 = @async  a(BIN, Z, m1, sd1, m2, sd2)
@time r = simP2st2Means(BIN, Z, m1, sd1, m2, sd2, delta; log10n=6)
print(r)
#println(fetch(p1))
#println(fetch(p2))
#println(fetch(p3))
