import cv2

class GaussianFilter():
    """
    GaussianFilter, donot change the solve function. You may add any other class 
    functions or other functions in the colab file. but refrain for function/class
    definitions already given. These will be used to grade later on.
    """

    def solve(self, img, ksize=(5,5), sigma_x=0):
        """
        Solve function to perform gaussian filtering.

        :param img: noisy image
        :param ksize: representing the size of the kernel.
        :param sigma_x: standard deviation in X direction
        :rtype: uint8 (w,h)
        :return: solved image
        """
        # [TODO] Can use cv2 inbuilt 
        Oimage = cv2.GaussianBlur(img, ksize, sigma_x, cv2.BORDER_DEFAULT)
        return Oimage
