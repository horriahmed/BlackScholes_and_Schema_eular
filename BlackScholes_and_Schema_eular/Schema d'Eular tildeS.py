import numpy as np
import numpy.random as sim
import matplotlib.pyplot as plt

T=4
n=1000
N=1000 # nombre de trajectoirs
pas =T/n

S0=10
r=0.2


S=np.ones((n+1,N))*S0
TildeS=np.ones((n+1,N))*S0

payoffAct=[]
k=S0

def callFunction(x,k):
    return max(x-k,0)


def sigma(t,x):
    y=0.1*(1+t/(1+x**2))
    return y;


for j in range(N):
    for i in range(1,n+1):
        TildeS[i,j]=TildeS[i-1,j]+sigma(pas*(i-1),TildeS[i-1,j])*TildeS[i-1,j]\
        *np.sqrt(pas)*sim.randn()
        S[i,j]=TildeS[i,j]*np.exp(r*pas*i)
    #integ=pas*np.sum(S[:j])
    payoffAct.append(np.exp(-r*T)*callFunction(S[n,j],k))

V0approx=np.mean(payoffAct)
print("Valeur Approximative du prix",V0approx)



#z c'est un pourcentage d'erreur la moyenne empirique converge vers E St

dates=np.linspace(0,T,n+1) # n+1 dates
graph=plt.plot(dates,S)
plt.show()