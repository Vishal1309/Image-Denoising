import matplotlib.pyplot as plt

def show_gray(img, title=""):
    """
    Function to show grayscale image
    """
    plt.imshow(img,cmap='gray')
    plt.title(title)