import cv2
import numpy as np

IMAGE = '../images/lena.bmp'
IMAGE_GRAY = '../images/lena.gray.bmp'

BLOCK_SIZE = 8

QUANTIZATION_MATRIX = np.array([
 [16, 11, 10, 16, 24, 40, 51, 61],
 [12, 12, 14, 19, 26, 58, 60, 55],
 [14, 13, 16, 24, 40, 57, 69, 56],
 [14, 17, 22, 29, 51, 87, 80, 62],
 [18, 22, 37, 56, 68, 109, 103, 77],
 [24, 35, 55, 64, 81, 104, 113, 92],
 [49, 64, 78, 87, 103, 121, 120, 101],
 [72, 92, 95, 98, 112, 100, 103, 99]
])

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


def split_blocks(img, block_size=8):
  """Returns 8x8 non overlapping blocks of an image as references to the original one"""
  rows, cols = img.shape
  return [img[i:i+block_size, j:j+block_size] for i in range(0,rows,block_size) for j in range(0,cols,block_size)]


def assemble_blocks(blocks, shape, block_size=8):
  """Assembles 8x8 non overlapping blocks into a single image"""
  newimg = np.zeros(shape, np.uint8)
  rows, cols = shape

  indices = [(i,j) for i in range(0, rows, block_size) for j in range(0, cols, block_size)]

  for block, index in zip(blocks, indices):
    i, j = index
    newimg[i:i+block_size, j:j+block_size] = block

  return newimg


def quantize(img, factor=1):
  """Quantizes an image using the quantization matrix multiplied by the specified factor"""
  matrix = QUANTIZATION_MATRIX * factor
  quantized = np.round(img / matrix) * matrix
  return quantized


def jpeg_block(block, quantization=1):
  """Applies JPEG compression to an 8x8 block and reverts it"""
  dct = cv2.dct(np.float32(block))
  quantized = quantize(dct, quantization)
  reverted = cv2.dct(np.float32(quantized), None, cv2.DCT_INVERSE)
  return np.round(reverted)


def jpeg_channel(channel, quantization=1):
  """Applies JPEG compression to an image channel and reverts it"""
  blocks = split_blocks(channel, BLOCK_SIZE)
  newblocks = [jpeg_block(block, quantization) for block in blocks]
  return assemble_blocks(newblocks, channel.shape)


def jpeg_image(ycc, factors):
  """Applies JPEG compression to an image in YCrCb with the specified quantization factors per channel and reverts it"""
  jpeged = [jpeg_channel(channel, cf) for channel, cf in zip(cv2.split(ycc), factors)]
  newycc = cv2.merge(jpeged)
  newimg = cv2.cvtColor(newycc, cv2.COLOR_YCR_CB2RGB)
  show_image('Compressed with YCrCb factors ' + str(factors), newimg)


def main_jpeg():
  """Applies JPEG compression to an image in RGB and reverts it"""
  img = cv2.imread(IMAGE)
  show_image('Original', img)

  ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)

  jpeg_image(ycc, [1,1,1])
  jpeg_image(ycc, [1,8,8])
  jpeg_image(ycc, [8,8,8])



if __name__ == '__main__':
  main_jpeg()
