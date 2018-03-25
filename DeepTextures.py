import glob
import sys
import os
from collections import OrderedDict
import numpy as np
import scipy

import caffe

from DeepImageSynthesis import ImageSyn
from DeepImageSynthesis import Misc
from DeepImageSynthesis import LossFunctions


import argparse

globalStep = 0


def print_progress():
    global globalStep 
    print("step : " + str(globalStep) )
    globalStep += 1

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help = "path to input image", type = str, required=True)
parser.add_argument('-o', '--output', help = "path to output image", type = str, required=True)
parser.add_argument('-m', '--maxiter', help = "Maximum number of iterations to perform",  type = int, default = 2000)

parser.add_argument('-g', '--gpu', help='use gpu', action='store_true')



args = parser.parse_args()
print(args)


if args.gpu:
	gpu = 0
	caffe.set_mode_gpu() #for cpu mode do 'caffe.set_mode_cpu()'
	caffe.set_device(gpu)

else:
	caffe.set_mode_cpu()


VGGweights = 'Models/vgg_normalised.caffemodel'
VGGmodel =  'Models/VGG_ave_pool_deploy.prototxt'

im_size = 256.
imagenet_mean = np.array([ 0.40760392,  0.45795686,  0.48501961]) #mean for color channels (bgr)



#load source image
source_img_org = caffe.io.load_image(args.input)

[source_img, net] = Misc.load_image(args.input, im_size, VGGmodel, VGGweights, imagenet_mean,  show_img=True)

im_size = np.asarray(source_img.shape[-2:])



#l-bfgs parameters optimisation
m = 20

#define layers to include in the texture model and weights w_l
tex_layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1']
tex_weights = [1e9,1e9,1e9,1e9,1e9]

#pass image through the network and save the constraints on each layer
constraints = OrderedDict()
net.forward(data = source_img)
for l,layer in enumerate(tex_layers):
    constraints[layer] = Misc.constraint([LossFunctions.gram_mse_loss],
                                    [{'target_gram_matrix': Misc.gram_matrix(net.blobs[layer].data),
                                     'weight': tex_weights[l]}])
    
#get optimisation bounds
bounds = Misc.get_bounds([source_img],im_size)

#generate new texture
result = ImageSyn.ImageSyn(net,  constraints, 
				  bounds=bounds,
				  callback=lambda x: print_progress(), 
                  minimize_options=
                      {
                      'maxiter': args.maxiter,
                       'maxcor': m,
                       'ftol': 0, 
                       'gtol': 0
                       }
                   )




#match histogram of new texture with that of the source texture and show both images
new_texture = result['x'].reshape(*source_img.shape[1:]).transpose(1,2,0)[:,:,::-1]
new_texture = Misc.histogram_matching(new_texture, source_img_org)


scipy.misc.imsave(args.output, new_texture)
