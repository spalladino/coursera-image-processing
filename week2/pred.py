import cv2
import numpy as np

IMAGE = '../images/lena.bmp'
IMAGE_GRAY = '../images/lena.gray.bmp'


def show_images(images):
  """Displays a set of pairs (image, title) one after the other"""
  for image, title in images:
    cv2.imshow(str(title or 'Image'), image)
    cv2.waitKey(0)

  cv2.destroyAllWindows()


def show_image(image, title="Image"):
  """Displays an image"""
  cv2.imshow(str(title), image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def show_hist(img, title='Histogram'):
  """Shows histogram of an image"""
  hist = cv2.calcHist([img], [0], None, [256], [0,256])
  cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
  hist = np.int32(np.around(hist))
  h = np.zeros((300,256,3))
  for x,y in enumerate(hist):
    cv2.line(h,(x,0),(x,y),(255,255,255))
  y = np.flipud(h)
  show_image(y, title)


def main_pred():
  # TODO: Implemente predictive compression
  img = cv2.imread(IMAGE_GRAY, 0)
  show_hist(img, 'Lena')
  show_image(img, 'Lena')


if __name__ == '__main__':
  main_pred()
