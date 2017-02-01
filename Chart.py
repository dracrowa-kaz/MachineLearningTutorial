import scipy as sp
import matplotlib.pyplot as plt


def error(f, x, y):
    return sp.sum((f(x)-y)**2)

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
print(data)
print(data.shape)
print(data.ndim)
x = data[:,0]
y = data[:,1]
print(sp.sum(sp.isnan(y)))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()

fp1, residuals, rank, sv, rcond = sp.polyfit(x,y,1,full=True)

print("Model parameters: %s" % fp1)
print(residuals)

fx = sp.linspace( 0, x[-1]+ 500, 1000)
f1 = sp.poly1d(fp1)
plt.plot(fx, f1(fx), linewidth=1)
#plt.legend(["d=%i" % f1.order], loc="upper left")

f2p = sp.polyfit(x, y, 2)
f2 = sp.poly1d(f2p)
print(error(f2,x,y))
#plt.plot(fx, f2(fx), linewidth=4)
#plt.legend(["d=%i" % f2.order], loc="upper left")

#Get data after 3.5 weeks.
inflextion = 3.5 * 7 * 24
xb = x[inflextion:]
yb = y[inflextion:]
fb = sp.poly1d(sp.polyfit(xb,yb,3))
plt.plot(xb,fb(xb))
fb2 = sp.poly1d(sp.polyfit(xb,yb,1))
plt.plot(xb,fb2(xb))

from scipy.optimize import fsolve
reached_max = fsolve(fb - 100000,100)/(7*24)
print(reached_max[0])


