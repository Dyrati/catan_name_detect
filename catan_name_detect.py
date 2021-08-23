import pytesseract
from PIL import Image, ImageGrab
import cv2 as cv
import numpy as np
import threading
import math
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def show(im):
    def inner():
        if isinstance(im, np.ndarray): Image.fromarray(im).show()
        elif isinstance(im, str): Image.open(im).show()
        else: im.show()
    t = threading.Thread(target=inner)
    t.start()

def clipboard():
    return ImageGrab.grabclipboard()

def aspect_ratio(points):
    xlist, ylist = points[:,0], points[:,1]
    xmin, xmax, ymin, ymax = min(xlist), max(xlist), min(ylist), max(ylist)
    return (ymax-ymin) / (xmax-xmin) if xmax != xmin else 0

def findcontours(im, threshold=60):
    if not isinstance(im, np.ndarray): im = np.array(im)
    imgray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(imgray, threshold, 255, 0)
    return cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]

def showcontours(im, threshold=60, index=-1, thickness=1):
    if not isinstance(im, np.ndarray): im = np.array(im)
    c = findcontours(im, threshold)
    im = im.copy()
    cv.drawContours(im, c, index, (0,0,0), thickness)
    show(im)

def findcontourindex(contours, coords):
    coords = np.array(coords)
    for i,contour in enumerate(contours):
        if any(all(coords == c) for c in contour[:,0]): return i

def isbanner(contour):
    contour = contour[:,0]  # contours are lists of lists of points (idk why), this just makes it a list of points
    if abs(aspect_ratio(contour)-0.2) > 0.05: return False
    start = min(enumerate(contour), key=lambda x: x[1][0])[0]  # make it start with leftmost rather than topmost point
    contour = np.concatenate((contour[start:], contour[:start+1]))  # make it complete the loop by including start twice
    rad2deg = 180/math.pi
    vectors = []
    for (x1,y1),(x2,y2) in zip(contour[:-1], contour[1:]):
        distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
        if distance <= 2: continue  # ignore 1-2 pixel direction changes
        angle = math.atan2(y2-y1, x2-x1)*rad2deg
        vectors.append((distance, angle))
    if not vectors: return False
    # combine adjacent segments with the same angle
    reduced = [vectors[0]]
    for distance, angle in vectors[1:]:
        dprev, aprev = reduced[-1]
        if angle == aprev:
            reduced[-1] = (distance + dprev, angle)
        else:
            reduced.append((distance, angle))
    angles = [a for d,a in reduced]
    i = 0
    for a in [90, 0, -135, -45, 180]:
        try: i += angles[i:i+5].index(a) + 1  # add leeway for short direction detours
        except ValueError: return False
    return True

def banner_contours(im):
    contours = findcontours(im)
    banners = [c[:,0] for c in reversed(contours) if isbanner(c)] # reversed to search top-down
    for i,b in enumerate(banners):
        start = min(enumerate(b), key=lambda x: x[1][0])[0]
        banners[i] = np.concatenate((b[start:], b[:start]))
    return banners

def polarize_gray(imgray):  # returns black and white image, preserving inbetween shades
    cmin, cmax = np.amin(imgray), np.amax(imgray)
    if cmin == cmax: return imgray
    imgray = (imgray.astype(np.uint32)-cmin)*255//(cmax-cmin)
    return imgray.astype(np.uint8)

def name_images(im):
    imgray = cv.cvtColor(np.array(im), cv.COLOR_RGB2GRAY)
    names = []
    for contour in banner_contours(im):
        x1,y1 = contour[0]
        x2,y2 = map(max, zip(*contour))
        fill_color = np.bincount(imgray[y1:y2, x1:x2].flatten()).argmax()
        # remove extraneous rows and columns from banner to help the AI
        rows = lambda matrix: enumerate(matrix)
        cols = lambda matrix: enumerate(matrix.transpose())
        rows_r = lambda matrix: enumerate(reversed(matrix))
        cols_r = lambda matrix: enumerate(reversed(matrix.transpose()))
        copy = (x1,y1,x2,y2)
        y1 += next((y for y,row in rows(imgray[y1:y2, x1:x2]) if np.count_nonzero(row == fill_color) >= len(row)//2), 0)
        y2 -= next((y for y,row in rows_r(imgray[y1:y2, x1:x2]) if np.count_nonzero(row == fill_color) >= len(row)//2), 0)
        x2 -= next((x for x,col in cols_r(imgray[y1:y2, x1:x2]) if all(col == fill_color)), 0)
        x2 -= next((x for x,col in cols_r(imgray[y1:y2, x1:x2]) if not all(col == fill_color)), 0) - 5
        x1 = x2 - next((x for x,col in cols_r(imgray[y1:y2, x1:x2]) if not any(col == fill_color)), x2-x1+1) + 1
        if x2-x1 <= 0 or y2-y1 <= 0: x1,y1,x2,y2 = copy
        polarized = polarize_gray(imgray[y1:y2, x1:x2])
        names.append(polarized)
    return names

def concatfill(arrays, fill=0, axis=0):
    arrays = list(arrays)
    maxdim = [max(x) for x in zip(*(a.shape for a in arrays))]
    for i,size in enumerate(maxdim):
        if i == axis: continue
        for j,a in enumerate(arrays):
            dim = list(a.shape)
            dim[i] = size-a.shape[i]
            padding = np.full(dim, fill)
            arrays[j] = np.concatenate((a, padding), axis=i)
    return np.concatenate(arrays, axis=axis)

def get_names(im):
    names = [pytesseract.image_to_string(img) for img in name_images(im)]
    for i,n in enumerate(names):
        match = re.search(r"[a-zA-Z0-9]+(?:#\d+)?", n[:-1].strip().replace(" ",""))
        names[i] = match.group() if match else ""
    return names