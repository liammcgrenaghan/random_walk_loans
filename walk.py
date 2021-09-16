
# import relevant libraries

import numba
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation, PillowWriter


accnt_d = {'F':[1,0,0,0],'B':[0,1,0,0],'G':[.1,0,.8,.1],'A':[.1,.1,.4,.4]}

# 'G' good standing accounts random walk
# generate df of a random walk, and the final absorbing states enter as a result of walking 
# through the markov chain

df_g = pd.DataFrame()

for j in range(1,101):
    
    absrb_accnt_type_list = []
    absrb_accnt_time_list = []
    
    sim_num = 'S'+ str(j)
    sim_num_time = 'S' + str(j) +'t'
    
    for i in range(1,101):

        accnt_type      = 'G'
        time_step = 0

        while accnt_type not in ['F','B']:
            upd_dist   = accnt_d[accnt_type[0]]
            accnt_type = random.choices(list(accnt_d.keys()), upd_dist, k=1)[0]
            print(accnt_type)
            time_step += 1

        #print(i)
        absrb_accnt_type_list.append(accnt_type)
        absrb_accnt_time_list.append(time_step)


    df_g[sim_num] = absrb_accnt_type_list
    df_g[sim_num_time] = absrb_accnt_time_list
    
d_rlln =  {} #pd.DataFrame() dont assign immediately to a df as there a performance improvments when assigning to a dictionary

sim_name_af = 'f_accnt_r_avg'
sim_name_ab = 'b_accnt_r_avg'
sim_name_time = 'time_r_avg'

# generate df containing the rolling frequencies of occurence of absorbing accounts and time taken to enter them

for z,x in enumerate(np.arange(200).reshape((100, 2))):  
    af_vec     = []
    ab_vec     = []

    t_vec      = []
    t_rlln_avg = []

    for a,t in zip(list(df_g.iloc[:,x[0]]),list(df_g.iloc[:,x[1]])):
        if a == 'F':
            af_vec.append(1)
            ab_vec.append(0)
        else:
            af_vec.append(0)
            ab_vec.append(1)

        t_vec.append(t)
        t_rlln_avg.append(np.mean(t_vec)) 

    af_rlln_vec = []
    af_rlln_avg = []

    ab_rlln_vec = []
    ab_rlln_avg = []

    for f,b in zip(af_vec, ab_vec):
        af_rlln_vec.append(f)
        af_rlln_avg.append(sum(af_rlln_vec)/len(af_rlln_vec))

        ab_rlln_vec.append(b)
        ab_rlln_avg.append(sum(ab_rlln_vec)/len(ab_rlln_vec))

    # now we have rolling average of F account,
    # B accounts and Time until an absorbing state is reached.

    df_rlln[sim_name_af + str(z)]   = af_rlln_avg
    df_rlln[sim_name_ab + str(z)]   = ab_rlln_avg
    df_rlln[sim_name_time + str(z)] = t_rlln_avg

df_rlln = pd.DataFrame(d_rlln)

dfplt = df_rlln.iloc[:,[0,1,2]]

# generate plot for single simulation
plt.rcParams['figure.figsize'] = [15, 7.5]

x = np.arange(1,101)
plt.ylim(-0.1, 1.1)
yf = np.array(dfplt.iloc[:,0])
yb = np.array(dfplt.iloc[:,1])
plt.plot(x, yf, '-o',ms=1.5, alpha=0.5, label='F')
plt.plot(x,yb, '-o', ms=0.5, alpha=0.5, color = 'red', label='B')

plt.ylabel("% split between absorbing states 'F' and 'B'")
plt.xlabel("Number of accounts put through the Markov Chain until absorbing state reached")

plt.axhline(y=0.875, color='b', linestyle='--',label='analytical F')
plt.axhline(y=0.125, color='r', linestyle='--',label='analytical B')

plt.legend()



#plt.savefig('/Users/liammcgrenaghan/sim1.png')
# save the figure
#plt.savefig('/Users/liammcgrenaghan/plot2.png', dpi=300, bbox_inches='tight')


plt.show()


# generate plot for all simulations 
%matplotlib inline
x = np.arange(1,101)



figure(figsize=(15, 7.5), dpi=100)

for a in np.arange(300).reshape((100, 3)):
    
    dfplt = df_rlln.iloc[:,[a[0],a[1],a[2]]]
    
    yf = np.array(dfplt.iloc[:,0])
    yb = np.array(dfplt.iloc[:,1])
    plt.plot(x, yf, '-o',linewidth=0.5, ms=1.5, alpha=0.3, color = 'blue')
    plt.plot(x, yb, '-o',linewidth=0.5, ms=1.5, alpha=0.3, color = 'red')

plt.ylabel("% split between absorbing states 'F' and 'B'")
plt.xlabel("Number of accounts put through the Markov Chain until absorbing state reached")

plt.axhline(y=0.875, color='b',linewidth=2.5, linestyle='--',label='analytical F')
plt.axhline(y=0.125, color='r',linewidth=2.5, linestyle='--',label='analytical B')
plt.plot(x, yf, '-o',linewidth=0.5, ms=1.5, alpha=0.3, color = 'blue',label='F')
plt.plot(x, yb, '-o',linewidth=0.5, ms=1.5, alpha=0.3, color = 'red',label='B')

plt.ylim(-0.1, 1.1)


plt.legend()


#plt.savefig('/Users/liammcgrenaghan/plot3.png', dpi=300, bbox_inches='tight')

plt.show()

x = np.arange(1,101)

dfplt_test = df_rlln.iloc[:,[0,1,2]]

# animate single simulation

# add random points for each line
%matplotlib notebook

 
# subplots() function you can draw
# multiple plots in one figure
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
 
# set limit for x and y axis
axes.set_ylim(-0.1, 1.1)
axes.set_xlim(0, 100)
 
plt.ylabel("% split between absorbing states 'F' and 'B'")
plt.xlabel("Number of accounts put through the Markov Chain until absorbing state reached")

 # after every iteration
x1, y1, y2 =  [], [], []

 
def animate(i):
    x1.append(x[i])
    y1.append(dfplt_test.iloc[:,0][i])
    y2.append(dfplt_test.iloc[:,1][i])

 
    axes.plot(x1, y1, color="red",label='F')
    axes.plot(x1, y2, color="gray",label='B')
    
 
# set ani variable to call the
# function recursively
anim = FuncAnimation(fig, animate, interval=5,frames=np.arange(0, 100, 1),blit=True, repeat=False)
#fig.canvas.draw()
#anim.event_source.stop()
#anim.save("/Users/liammcgrenaghan/anim4.gif")



# animate all simulations at once
%matplotlib notebook

 
# subplots() function you can draw
# multiple plots in one figure
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
 
# set limit for x and y axis
axes.set_ylim(-0.1, 1.1)
axes.set_xlim(0, 100)
 
plt.ylabel("% split between absorbing states 'F' and 'B'")
plt.xlabel("Number of accounts put through the Markov Chain until absorbing state reached")

 # after every iteration
x1, y1, y2 =  [], [], []

 
def animate(i):
    #x1.append(x[i])
    
    for k in np.arange(300).reshape(100,3):
        
        dfplt_test = df_rlln.iloc[:,k]
        
        x1.append(x[i])
        y1.append(dfplt_test.iloc[:,0][i])
        y2.append(dfplt_test.iloc[:,1][i])
        
        axes.plot(x1, y1, color="red",label='F')
        axes.plot(x1, y2, color="gray",label='B')
    
    
 
 
# set ani variable to call the
# function recursively
anim = FuncAnimation(fig, animate, interval=5,frames=np.arange(0, 100, 1),blit=True, repeat=False)
fig.canvas.draw()
anim.event_source.stop()
anim.save("/Users/liammcgrenaghan/anim5.gif")