import png
import cudapy as cp
from time import clock


# mandel : int (float, float, int)
def mandel(c_re, c_im, count):
  z_re = c_re
  z_im = c_im

  for i in xrange(0, count):
    if z_re * z_re + z_im * z_im > 4.0:
      break
    new_re = z_re * z_re - z_im * z_im
    new_im = 2.0 * z_re * z_im
    z_re = c_re + new_re
    z_im = c_im + new_im

  return i


# mandelbrot : void (float, float, float, float, int, int, int, int*)
def mandelbrot(x0, y0, x1, y1, width, height, maxIter, output):
  if idx >= width or idy >= height:
    return
  i = idy * width + idx

  dx = (x1 - x0) / float(width)
  dy = (y1 - y0) / float(height)

  x = x0 + float(idx) * dx
  y = y0 + float(idy) * dy
  output[i] = mandel(x, y, maxIter)


__mandelbrotCall = cp.compile([mandelbrot, mandel])

def mandelbrotCall(x0, y0, x1, y1, width, height, maxIter):
  cudaResult = cp.CudaArray.allocate(width * height, cp.Int)
  __mandelbrotCall(width, height)(x0, y0, x1, y1, width, height, maxIter, cudaResult)
  return cudaResult.toHost()


def scaleAndShift(x0, y0, x1, y1, scale, shiftX, shiftY):
  x0 *= scale
  x1 *= scale
  y0 *= scale
  y1 *= scale
  x0 += shiftX
  x1 += shiftX
  y0 += shiftY
  y1 += shiftY
  return (x0, y0, x1, y1)


def savePng(f, raw, width, height):
  raw = map(lambda x : ((x / 256.0) ** 0.5) * 255.0 , raw)
  rows = [raw[i * width : i * width + width] for i in xrange(height)]
  with open(f, 'wb') as f: # binary mode is important
    w = png.Writer(width, height, greyscale=True)
    w.write(f, rows)


if __name__ == "__main__":
  width, height, maxIters = 1200, 800, 256
  x0, y0, x1, y1 = -2.0, -1.0, 1.0, 1.0
  if len(sys.argv) >= 2 and sys.argv[1] == "2":
    x0, y0, x1, y1 = scaleAndShift(x0, y0, x1, y1, 0.015, -0.986, 0.3)

  start = clock()
  raw = mandelbrotCall(x0, y0, x1, y1, width, height, maxIters)
  total = clock() - start
  print "Width:", width, "height:", height, "maxIters:", maxIters
  print "Time:", total

  savePng("mandel.png", raw, width, height)
