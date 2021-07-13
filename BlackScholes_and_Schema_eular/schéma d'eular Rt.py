import numpy as np
import numpy.random as sim
import matplotlib.pyplot as plt

T=3
n=100
N=1 # nombre de trajectoirs
pas =T/n


alpha=0.75; theta=0.015; beta=0.15;
r0=0.02

r=r0*np.ones((n+1,N))
interg=np.zeros((n+1,N))
coeffAct=np.ones((n+1,N))
B=[1];R=[]

S0=50
TildeS=S0*np.ones((n+1,N))
S=S0*np.ones((n+1,N))
PayoffAct=[]
MoyS=[]
k=S0/2

def sigma(t,x):
    y=0.2*np.sin(2*np.pi*t/T)+t+1/(1+x**2)
    return y
def call(x,k):
    return np.max(x-k,0)

for j in range(N):
    for i in range(1,n+1):
        r[i,j]=r[i,j-1]+alpha*(theta-r[i-1,j])*pas+beta*np.sqrt(abs(r[i-1,j]))*np.sqrt(pas)*sim.randn()# sim.randn pour simmuler Gi
        interg[i,j]=pas*np.sum(r[:i-1,j])
        coeffAct[i,j]=np.exp(-1*interg[i,j])
        TildeS[i,j]=TildeS[i-1,j]+sigma(pas*(i-1),TildeS[i-1,j])*TildeS[i-1,j]*np.sqrt(pas)*sim.randn()
        S[i,j]=TildeS[i,j]*np.exp(interg[i,j])
    MoyS.append(np.mean(S[:,j]))
    aux=call(MoyS[-1],k)
    aux1=aux*coeffAct[n,j]
    PayoffAct.append(aux)

prix =np.mean(PayoffAct)
print("prix =",prix)

for i in range(1,n+1):
    b=np.mean(coeffAct[i,:])
    B.append(b)
    R.append(b**(-1/(pas*i))-1)

R.append(R[-1])

dates=np.linspace(0,T,n+1)
graph=plt.plot(dates, R)
plt.show()
