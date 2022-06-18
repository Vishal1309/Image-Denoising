from tqdm import tqdm
import numpy as np
import cv2

def get_window(img, x, y, N=25):
  h, w, c = img.shape             # Extracting the dimensions of the image

  dist = N//2                      # Dist from center to get window
  window = np.zeros((N, N, c))
 
  xmin = max(0, x-dist)
  xmax = min(w, x+dist+1)
  ymin = max(0, y-dist)
  ymax = min(h, y+dist+1)

  window[dist - (y-ymin):dist + (ymax-y), dist - (x-xmin)
                :dist + (xmax-x)] = img[ymin:ymax, xmin:xmax]

  return window

class NLMeans():
  """
  Non Local Means, donot change the solve function. You may add any other class 
  functions or other functions in the colab file. but refrain for function/class
  definitions already given. These will be used to grade later on.
  """
  def example(self,img,**kwargs):
    denoised_image = cv2.fastNlMeansDenoising(img,**kwargs)
    return denoised_image

  def flatten_neigh(self, neigh_mat, x, y):
    return neigh_mat[y,x].flatten()

  def Add_weights(self, F):
    return np.sum(F)

  def solve(self,img,h=30,small_window=7,big_window=21):
    """
    Solve function to perform nlmeans filtering.

    :param img: noisy image
    :param h: sigma h (as mentioned in the paper)
    :param small_window: size of small window
    :param big_window: size of big window
    :rtype: uint8 (w,h)
    :return: solved image
    """
    t = (big_window - 1)//2
    f = (small_window - 1)//2
    # size of the smaller neighbourhood
    N = small_window

    # sliding window size (the bounding space)
    S = big_window

    # Filtering Parameter
    sigma_h = h

    # Padding the image
    pad_img = np.pad(img, t+f)

    h_pad, w_pad = pad_img.shape
    h, w = img.shape

    neigh_mat = np.zeros((h+S-1, w+S-1, N, N))

    # Making a neighbourhood for all the pixels for vectorizing sliding window algorithm
    for y in range(h+S-1):
      for x in range(w+S-1):
        neigh_mat[y, x] = np.squeeze(get_window(pad_img[:, :, np.newaxis], x+f, y+f, 2*f+1))

    # Creating an empty image to be filled by the algorithm
    res = np.zeros(img.shape)

    # Initializing counter
    prog = tqdm(total=(h-1)*(w-1), position=0, leave=True)

    # Iterating over every pixel in the image
    for Y in range(h):
      for X in range(w):
        # taking padding into consideration (shifting)
        x = X + t
        y = Y + t
        # Getting neibourhood of pixel in chunks of search window
        Window = get_window(np.reshape(neigh_mat, (h+S-1, w+S-1, N*N)), x, y, S)
        
        # Getting the self Neigbourhood
        self_neigh = self.flatten_neigh(neigh_mat, x, y)

        # Distance of vectorized neibourhood
        distance = Window-self_neigh

        # Finding weights
        d_sq = distance*distance
        e = np.sqrt(np.sum(d_sq, axis=2))
        F = np.exp(-e/(sigma_h*sigma_h))

        # Adding weights
        Z = self.Add_weights(F) # normalizing factor

        # Computing average pixel value
        img_part = np.squeeze(get_window(pad_img[:, :, None], x+f, y+f, S))

        NL = np.sum(F*img_part)
        
        #normalizing
        res[Y, X] = NL/Z

    return res

# [TODO]
# Will be checked on image 3.
# Report your best salt_and_paper_h: 10
# Report your best gaussian_h: 9