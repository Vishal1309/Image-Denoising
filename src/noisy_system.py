from skimage.util import random_noise
import numpy as np
import random
import cv2

class noisy_system():
  def example(self,img,**kwargs):
    """
    An example function to test expected return.
    You can read more about skimage.util.random_noise at https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
    """
    noisy_image = random_noise(img,**kwargs)
    noisy_image = np.uint8(noisy_image*255)
    return noisy_image

  def create_salt_and_pepper_noise(self,img,amount=0.05):
    """
    function to create salt and pepper noise
    :param image: input image
    :rtype: uint8 (w,h)
    :return: noisy image
    
    """
    # [TODO]
    noisy_pixels=(img.shape[0]*img.shape[1])*0.025

    # adding salt
    for i in range(int(noisy_pixels)):
      x=random.randint(0,img.shape[0]-1)
      y=random.randint(0,img.shape[1]-1)
      img[x][y]=255

    # adding pepper
    for i in range(int(noisy_pixels)):
      x=random.randint(0,img.shape[0]-1)
      y=random.randint(0,img.shape[1]-1)
      img[x][y]=0

    res =np.uint8(img)
    return res

  def create_gaussian_noise(self,img,mean=0,var=0.01):
    """
    function to create gaussian noise
    :param image: input image
    :rtype: uint8 (w,h)
    :return: noisy image
    """
    # [TODO]
    
    gaussian_noise = np.random.normal(mean,np.sqrt(var),(img.shape[0],img.shape[1]))
    gaussian_noise = gaussian_noise.reshape(img.shape[0],img.shape[1])
    gaussian_noise*=255
    res = img + gaussian_noise
    return res