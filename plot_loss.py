import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


#plt.rcParams["font.family"] = "Helvetica Neue"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex = True)
plt.rcParams.update({'font.size': 16})

file = open("circles.txt", 'r')
for line in file.readlines():
    fname = line.rstrip().split(',')
loss = list(map(float, fname))

file = open("moons.txt", 'r')
for line in file.readlines():
    fname = line.rstrip().split(',')
loss1 = list(map(float, fname))


file = open("blobs.txt", 'r')
for line in file.readlines():
    fname = line.rstrip().split(',') 
loss2 = list(map(float, fname))

fig = plt.figure(figsize=(9, 5))
plt.plot(loss, label='circles')
plt.plot(loss1, label='moons')
plt.plot(loss2, label='blobs')

plt.xlabel('training step iteration')
plt.ylabel('loss/cost')
plt.legend()

plt.savefig('losses.eps', format='eps', dpi=1000)

file = open("reuploadingcircles.txt", 'r')
for line in file.readlines():
    fname = line.rstrip().split(',')
loss = list(map(float, fname))

file = open("reuploadingmoons.txt", 'r')
for line in file.readlines():
    fname = line.rstrip().split(',')
loss1 = list(map(float, fname))


file = open("reuploadingblobs.txt", 'r')
for line in file.readlines():
    fname = line.rstrip().split(',') #using rstrip to remove the \n
loss2 = list(map(float, fname))

fig = plt.figure(figsize=(9, 5))
plt.plot(loss, label='circles')
plt.plot(loss1, label='moons')
plt.plot(loss2, label='blobs')

plt.xlabel('training step iteration')
plt.ylabel('loss/cost')
plt.legend()
#plt.show()
plt.savefig('lossesreuploading.eps', format='eps', dpi=1000)