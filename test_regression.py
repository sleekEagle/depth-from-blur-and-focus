import matplotlib.pyplot as plt
import numpy as np

def get_sim_blur(s1_start=0,s1_end=2,n=100,noise_std=1,s2=1.3):
    s1 = np.linspace(s1_start,s1_end,n)
    L=np.abs(s2-s1)
    noise = np.random.normal(0,noise_std,n)
    #add noise
    L+=noise
    return s1,L

def get_s2est(s1,L):
    #get the first estimates for s2
    s1est1=s1-L # for the +45 degrees line
    s1est2=s1+L # for the -45 degrees line

    vals=[]
    for i in range(len(s1est1)+1):
        tmp=[]
        for j in range(len(s1est1)):
            if(j>=i):
                tmp.append(s1est1[j])
            else:
                tmp.append(s1est2[j])
        vals.append(tmp)


    valstds=[]
    for i in range(len(vals)):
        valstds.append(np.std(vals[i]))
    minstdarg=np.argmin(valstds)
    S_estlist=vals[minstdarg]
    S_est=np.mean(S_estlist)

    return S_est

s2=0.3
s1,L=get_sim_blur(1.4,2.7,5,0.001,s2)

#plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(s1,L,marker='o',color='r')
#plt.show()

S_est=get_s2est(s1,L)
print(S_est)

























