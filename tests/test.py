from dubious import Normal, Beta, Uncertain, Context

#We can define distribution parameters with other distributions.
normal = Normal(10, 1)
beta = Beta(3,normal)
ctx1 = Context()
ctx2 = Context()

x = Uncertain(normal, ctx=ctx1)
y = Uncertain(beta, ctx=ctx2)

#Apply some arithmetic.
x = x*y

print(f"{x.mean()} : {x.var()}")



