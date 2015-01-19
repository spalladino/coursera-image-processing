import cv2
import numpy as np

IMAGE = '../images/lena.bmp'
IMAGE_GRAY = '../images/lena.gray.bmp'


def reduce_intensity(img, factor):
  """Quantizes an image by the specified factor"""
  return (img / factor) * factor


def show_images(images):
  """Displays a set of pairs (title, image) one after the other"""
  for title, image in images:
    cv2.imshow(str(title), image)
    cv2.waitKey(0)

  cv2.destroyAllWindows()


def show_image(title, image):
  """Displays an image"""
  cv2.imshow(str(title), image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def shrink_resolution(img, factor):
  """Reduces resolution of an image by reducing and re enlarging it using the average of the neighbouring pixels"""
  shrunk = cv2.resize(img, (0,0), None, 1.0/factor, 1.0/factor, cv2.INTER_AREA)
  return cv2.resize(shrunk, (0,0), None, factor, factor, cv2.INTER_AREA)


def main_shrink_resolution():
  """Replaces pixel boxes by their average by shrinking and re enlarging the image"""
  img = cv2.imread(IMAGE_GRAY)
  images = [(n, shrink_resolution(img, n)) for n in (3,5,7,20,100)]
  show_images(images)


def main_image_rotate():
  """Rotates the image"""
  img = cv2.imread(IMAGE_GRAY)
  rows, cols, _ = img.shape
  rotation = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
  rotated = cv2.warpAffine(img, rotation, (cols, rows))
  show_image('45', rotated)


def main_image_blur():
  """Applies a blur with different kernel sizes"""
  img = cv2.imread(IMAGE_GRAY)
  images = [(n, cv2.blur(img, (n,n))) for n in [3,10,20,100]]
  show_images(images)


def main_image_boxfilter():
  """Applies a box filter with different kernel sizes"""
  img = cv2.imread(IMAGE_GRAY)
  images = [(n, cv2.boxFilter(img, -1, (n,n))) for n in [3,10,20,100]]
  show_images(images)


def main_image_filter2d():
  """Applies a 2D filter with several nxn normalised kernels, ie nxn matrices of 1s"""
  img = cv2.imread(IMAGE_GRAY)
  images = [(n, cv2.filter2D(img, -1, np.ones((n,n),np.float32)/(n*n))) for n in [3,10,20,100]]
  show_images(images)


def main_image_intensities():
  """Displays intensities from original to just 2 colors"""
  img = cv2.imread(IMAGE_GRAY)
  images = [(2**exp, reduce_intensity(img, 2**exp)) for exp in range(0,8)]
  show_images(images)


def main_image_to_grayscale():
  """Converts image to grayscale and saves"""
  img = cv2.imread(IMAGE)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imwrite(IMAGE_GRAY, gray)
  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
  main_shrink_resolution()

