from dubious import Normal, Beta, Uncertain, Context, umath

normal = Normal(20, 4)
normal2 = Normal(10,2)

x = Uncertain(normal) + Uncertain(normal2)

print(f"variance: {x.var()} mean: {x.mean()} q(0.05): {x.quantile(0.05)}")



