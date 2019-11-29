using Convex
using ECOS
using Clp

x = Variable(4)

# objective
profit = 50*x[1]+40*x[2]+60*x[3]+30*x[4]

# Time constraint
a = [120, 100, 180, 140]
b = 5000

#########################
## NOMINAL FORMULATION ##
#########################

nominal_constrs = [
    x>=0,
    dot(a, x)<=b,
    2*x[1]+3*x[2]+6*x[3]+1*x[4] <= 80,
    3*x[1]+2*x[2]+2*x[3]+0*x[4] <= 40,
]

nominal_problem = maximize(profit, nominal_constrs)
solve!(nominal_problem, ClpSolver())

println("Optimal Nominal Solution: ", x.value)

########################
## ROBUST FORMULATION ##
########################

# Time constraint ellipse
E = diagm(0=>a)*0.3

robust_constrs = [
    x>=0,
    dot(a, x)+norm(sqrt(E)*x)<=b,
    2*x[1]+3*x[2]+6*x[3]+1*x[4] <= 80,
    3*x[1]+2*x[2]+2*x[3]+0*x[4] <= 40,
]

robust_problem = maximize(profit, robust_constrs)
solve!(robust_problem, ECOSSolver(verbose=false))

println("Optimal Robust Solution: ", x.value)

#############################
## LARGE SCALE FORMULATION ##
#############################

function wc_val(a, E, x)
    return a+E*x/sqrt(dot(x, E*x))
end

relaxed_constrs = nominal_constrs
for k in 1:20

    xk=x.value
    a_wc = wc_val(a, E, xk)
    global relaxed_constrs = relaxed_constrs + [dot(a_wc,x)<=b]
    relaxed_problem = maximize(profit, relaxed_constrs)
    solve!(relaxed_problem, ClpSolver())

    println("Relaxed Solution (iteration " * string(k) *") : "* string(x.value)) 
end
