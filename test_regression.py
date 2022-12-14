import matplotlib.pyplot as plt
import numpy as np

def get_sim_blur(s1_start=0,s1_end=2,n=100,noise_std=1,s2=1.3):
    s1 = np.linspace(s1_start,s1_end,n)
    L=np.abs(s2-s1)
    noise = np.random.normal(0,noise_std,n)
    #add noise
    L+=noise
    return s1,L

s2=1.3
s1,L=get_sim_blur(0,2,5,0.06,s2)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(s1,L,marker='o',color='r')
plt.show()


