import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001
filename = 'data10_points.txt'

with open(filename) as f_obj:
    N_POINTS = len(f_obj.readlines())

G = nn.Sequential(                      # Generator
    nn.Linear(3, 128),            # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(128, 3),     # making a painting from these random ideas
)

D = nn.Sequential(                      # Discriminator
    nn.Linear(3, 128),     # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

for step in range(100):
#    artist_paintings = artist_works()           # real painting from artist
    Real_pillar = np.loadtxt(filename)
    Real_pillar = torch.from_numpy(Real_pillar).float()

    N_ideas = torch.randn(N_POINTS,3)  # random ideas

    G_pillar = G(N_ideas)                    # fake painting from G (random ideas)

    prob_pillar0 = D(Real_pillar)          # D try to increase this prob
    prob_pillar1 = D(G_pillar)               # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_pillar0) + torch.log(1. - prob_pillar1))
    G_loss = torch.mean(torch.log(1. - prob_pillar1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
#
#     if step % 50 == 0:  # plotting
#         plt.cla()
#         plt.title("X-Y Scatter Diagram.")
#         plt.xlabel("x")
#         plt.ylabel("y")
#         #plt.xlim(-1,700)
#         #plt.ylim(-1,10)
#         plt.scatter(G_pillar[0],G_pillar[1],c='black',edgecolors=None,s=0.5)
#         plt.pause(0.01)
#
# plt.ioff()
# plt.show()

x_ordi = []
y_ordi = []
z_ordi = []
for i in range(N_POINTS):
    x_ordi.append(float(G_pillar[i][0]))
    y_ordi.append(float(G_pillar[i][1]))
    z_ordi.append(float(G_pillar[i][2]))

savefile = 'G_data10.txt'
with open(savefile,'w') as fw_obj:
    for i in range(N_POINTS):
        fw_obj.write(str(x_ordi[i])+" "+str(y_ordi[i])+" "+str(z_ordi[i])+"\n")
# plt.title("X-Y Scatter Diagram.")
# plt.xlabel("x")
# plt.ylabel("y")
# # plt.xlim(-1,700)
# # plt.ylim(-1,10)
# plt.scatter(x_ordi,y_ordi,c='black',edgecolors=None,s=0.5)
# plt.show()

