import numpy as np
from glob import glob
from image_helper_cycle_gan import ImageHelper
from cycle_gan import CycleGAN

image_helper = ImageHelper()

dataset_folder = "datasets/monet2photo"   #modify this folder

print("Ploting the images...")
filenames = np.array(glob(dataset_folder+'/testA/*.jpg'))
image_helper.plot20(filenames)

generative_advarsial_network = CycleGAN((128, 128, 3), 10.0, image_helper)
generative_advarsial_network.train(100, 1, dataset_folder)
