import numpy as np 
import scipy
import matplotlib.pyplot as plt 

# The function that maps z to x1 and x2
def f(z):
  x_1 = np.exp(np.sin(2+z*3.675)) * 0.5
  x_2 = np.cos(2+z*2.85)
  return x_1, x_2

def draw_3d_projection(z,pr_z, x1,x2):
  alpha = pr_z / np.max(pr_z)
  ax = plt.axes(projection='3d')
  fig = plt.gcf()
  fig.set_size_inches(5.5, 5.5)
  for i in range(len(z)-1):
    ax.plot([z[i],z[i+1]],[x1[i],x1[i+1]],[x2[i],x2[i+1]],'r-', alpha=pr_z[i])
  ax.set_xlabel('$z$',)
  ax.set_ylabel('$x_1$')
  ax.set_zlabel('$x_2$')
  ax.set_xlim(-3,3)
  ax.set_ylim(0,2)
  ax.set_zlim(-1,1)
  ax.set_box_aspect((3,1,1))
  plt.show()

# Compute the prior
def get_prior(z):
  return scipy.stats.multivariate_normal.pdf(z)

z = np.arange(-3.0,3.0,0.01)
# Find the probability distribution over z
pr_z = get_prior(z)

# plt.plot(z, pr_z)
# plt.show()

# Define the latent variable values
z = np.arange(-3.0,3.0,0.01)
# Find the probability distribution over z
pr_z = get_prior(z)
# Compute x1 and x2 for each z
x1,x2 = f(z)
# Plot the function
draw_3d_projection(z,pr_z, x1,x2)