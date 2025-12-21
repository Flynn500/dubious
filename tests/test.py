from dubious import Normal, Beta, Uncertain, Context

ctx = Context()

#We can define distribution parameters with other distributions.
normal = Normal(10, 1)
beta = Beta(3,normal)

x = Uncertain(normal, ctx=ctx)
y = Uncertain(beta, ctx=ctx)

#Apply some arithmetic.
x = x*y

print(x.sample(5))

#We can also use uncertain distributions to define parameters.
normal3 = Normal(y+2, 3)
print(normal3.mean())

