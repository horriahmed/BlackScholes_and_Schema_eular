import numpy as np
import numpy.random as sim
import matplotlib.pyplot as plt

T=4
n=1000
N=1000 # nombre de trajectoirs
pas =T/n

S0=10
sig=0.2
mu=0.08

S=np.ones((n+1,N))*S0

def F(t,x):
    y=S0*np.exp(sig*x +(mu-0.5*sig**2)*t)
    return y

B=np.zeros((n+1,N))
#Bt0=0 déja initialisé avec la matrice B.zero

for j in range(N):
    for i in range(1,n+1):
        B[i,j]=B[i-1,j]+np.sqrt(pas)*sim.randn()
        S[i,j]=F(pas*i,B[i,j])  # pas*i = T/N *i

z=0
for j in range (N):
    z=z+S[n,j]

z=(z/N )-S0*np.exp(mu*T)
z=z/S0*np.exp(mu*T)

print("Erreur Relative=",z)

#z c'est un pourcentage d'erreur la moyenne empirique converge vers E St

dates=np.linspace(0,T,n+1) # n+1 dates
graph=plt.plot(dates,S)
plt.show()