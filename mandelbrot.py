# Import libraries for simulation
import imageio as im
import tensorflow as tf
import numpy as np
# Imports for visualization
import PIL.Image
from os import remove
from io import BytesIO

images = []
for d in range(50):
	center = np.array([0.3,0])
	zoom = d + 1
	xLength = 2.6
	yLength = 3

	xRange = np.array([-xLength/2, xLength/2]) / zoom + center[0]
	yRange = np.array([-yLength/2, yLength/2]) / zoom + center[1]
	interval = 0.005 / zoom

	sess = tf.InteractiveSession()
	# Use NumPy to create a 2D array of complex numbers
	Y, X = np.mgrid[yRange[0]:yRange[1]:interval, xRange[0]:xRange[1]:interval]
	# X is grid of -1.3 to 1.3 (increments of 0.005),
	# Y is grid -2 to 1 (increments of 0.005)
	Z = X+1j*Y
	# Z is the array of complex numbers X[i][j]+j*Y[i][j]
	xs = tf.constant(Z.astype(np.complex64))
	# xs is a constant tensor of Z in complex64
	zs = tf.Variable(xs)
	# zs is a variable tensor equal to xs
	ns = tf.Variable(tf.zeros_like(xs, tf.float32))
	# ns is an empty variable tensor of the same dimensionality as xs,zs, but of type float32 rather than complex64
	tf.global_variables_initializer().run()
	# xs, zs, ns are now initialized
	# Compute the new values of z: z^2 + x
	zs_ = zs**2 + xs

	# Have we diverged with this new value?
	not_diverged = tf.abs(zs_) < 4

	# Operation to update the zs and the iteration count.
	#
	# Note: We keep computing zs after they diverge! This
	#       is very wasteful! There are better, if a little
	#       less simple, ways to do this.
	#
	step = tf.group(
	  zs.assign(zs_),
	  ns.assign_add(tf.cast(not_diverged, tf.float32))
	  )
	for i in range(200): step.run()

	def DisplayFractal(a, fmt='png'):
	  """Display an array of iteration counts as a
		 colorful picture of a fractal."""
	  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
	  img = np.concatenate([10+20*np.cos(a_cyclic),
							30+50*np.sin(a_cyclic),
							155-80*np.cos(a_cyclic)], 2)
	  img[a==a.max()] = 0
	  a = img
	  a = np.uint8(np.clip(a, 0, 255))
	  f = BytesIO()
	  PIL.Image.fromarray(a).save('mandelbrot'+str((d/10.0))+'.'+fmt, fmt)
	  images.append(im.imread('mandelbrot'+str((d/10.0))+'.'+fmt, fmt))
	  print(d)
	  remove('mandelbrot'+str((d/10.0))+'.'+fmt)

	DisplayFractal(ns.eval())

im.mimsave('mandelbrot.gif', images)