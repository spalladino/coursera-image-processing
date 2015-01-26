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


def calc_entropy(img):
  """Calculates entropy for a given image"""
  # See http://stackoverflow.com/a/16650551/12791
  hist = cv2.calcHist([img],[0],None,[256],[0,256])
  hist = hist.ravel()/hist.sum()
  logs = np.nan_to_num(np.log2(hist))
  entropy = -1 * (hist*logs).sum()
  return entropy


def pixels(img):
  """Yields (pixels, i, j) of an image in row order"""
  rows, cols = img.shape
  for i in range(rows):
    for j in range(cols):
      yield (img.item(i,j), i, j)


def predictive_compress(img, predictor):
  """Compresses an image using given predictor"""
  output = np.zeros(img.shape)
  for pixel, i, j in pixels(img):
    predicted = predictor(img, i, j)
    output.itemset(i, j, pixel - predicted)

  return np.uint8(np.round(output))


def predictive_uncompress(img, predictor):
  """Uncompresses an image using given predictor"""
  uncompressed = np.zeros(img.shape)
  for error, i, j in pixels(img):
    predicted = predictor(uncompressed, i, j)
    uncompressed.itemset(i, j, predicted + error)

  return np.uint8(np.round(uncompressed))


def predictor_previous_row(img, i, j):
  """Predicts pixel based on the value of (-1, 0)"""
  if i == 0: return 128
  return img.item(i-1, j)


def predictor_previous_col(img, i, j):
  """Predicts pixel based on the value of (0, -1)"""
  if j == 0: return 128
  return img.item(i, j-1)


def predictor_average(img, i, j):
  """Predicts pixel based on the value of the average of (0, -1), (-1, 0) and (-1, -1)"""
  neighbours = []
  if i != 0: neighbours.append(img.item(i-1, j))
  if j != 0: neighbours.append(img.item(i, j-1))
  if i != 0 and j != 0: neighbours.append(img.item(i-1, j-1))
  if len(neighbours) == 0: return 128
  return sum(neighbours)/len(neighbours)


def entropy_for_predictor(img, predictor):
  """Returns entropy for a given predictor"""
  compressed = predictive_compress(img, predictor)
  return calc_entropy(img)


def main_pred():
  img = cv2.imread(IMAGE_GRAY, 0)

  for predictor in [predictor_previous_row, predictor_previous_col, predictor_average]:
    compressed = predictive_compress(img, predictor)
    print "Entropy with {0}: {1}".format(predictor.__name__, str(calc_entropy(compressed)))
    show_hist(compressed + 128, predictor.__name__)


if __name__ == '__main__':
  main_pred()
