import sys
sys.path.append('/usr/local/python/2.7/')
import os
import cv2
import scipy.ndimage.filters as filters
import numpy as np
from matplotlib import pyplot as plt
import tesserocr
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
from pdf_bank import get_otsu


def get_contour_boxes(otsu, bbox_width, bbox_height):
    '''Get contours for binary image filtered by width and height'''

    contours, hierarchy = cv2.findContours(255 - otsu.copy(),
                                           cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_bbox_height = lambda h: bbox_height[0] <= h <= bbox_height[1]
    valid_bbox_width  = lambda w: bbox_width[0]  <= w <= bbox_width[1]
    return  filter(lambda (x, y, w, h): valid_bbox_height(h) and valid_bbox_width(w),
                   map(cv2.boundingRect, contours))


def apply_contour_mask(orig, otsu, extra_border=(0, 15),
                       bbox_width=(50, np.inf), bbox_height=(15, np.inf)):
    '''Clear areas not in contours'''
    
    clear_mask = np.ones(otsu.shape, dtype=np.bool)
    h_add, w_add = extra_border
    
    for (x, y, w, h) in get_contour_boxes(otsu, bbox_width, bbox_height):
        clear_mask[y-h_add:y+h+h_add, x-w_add:x+w+w_add] = False

    result = np.maximum(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), orig.max(axis=2))
    result[clear_mask] = 255
    return result


def apply_contour_cleaning(orig, otsu):
    '''Clean original image using contours and Otsu binary image'''
    clean = apply_contour_mask(orig, otsu)
    new_otsu = get_otsu(clean)
    return cv2.bitwise_or(clean, new_otsu), new_otsu


def get_cleaned_tabular_image(clean, evidence):
    cln = clean.copy()
    for items in evidence.itervalues():
        for (_, _, box) in items:
            x, y, w, h = box
            cln[y:y+h, x:x+w] = -1
    mask = np.empty(clean.shape, clean.dtype)
    mask[:] = -1
    for x, y, w, h in get_image_segments(cln):
        mask[y:y+h, x:x+w] = 0
    return cv2.bitwise_or(clean, mask)


def get_image_segments(img, threshold=223, v_kernel=50, h_kernel=20, width=2480):
    '''Extract image segments'''
    segments = []
    mask = img >= threshold
    h_profile = np.sum(mask, axis=1)
    jumps = np.nonzero(np.diff(filters.minimum_filter1d(h_profile, v_kernel) >= width))[0]
    if len(jumps) % 2 == 1:
        print 'Skipping. Jumps =', jumps
        return segments
    for y_start, y_end in jumps.reshape(-1, 2):
        height = y_end - y_start
        v_profile = np.sum(mask[y_start:y_end, :], axis=0)
        h_jumps = np.nonzero(np.diff(filters.minimum_filter1d(v_profile, h_kernel) >= height))[0]
        for x_start, x_end in h_jumps.reshape(-1, 2):
            width = x_end - x_start
            segments.append( (x_start, y_start, width, height) )
    return segments


def get_segment_text(img, segments=None, display=False):
    '''Extract image text by segments'''
    text = {}
    image = Image.fromarray(img)
    with PyTessBaseAPI(lang='swe') as api:
        api.SetImage(image)
        if segments is None:
            boxes = api.GetComponentImages(RIL.TEXTLINE, True)
            segments = map(lambda b: (b[1]['x'], b[1]['y'], b[1]['w'], b[1]['h']), boxes)
        for i, bbox in enumerate(segments):
            api.SetRectangle(*bbox)
            ocrResult = api.GetUTF8Text().strip()
            conf = api.MeanTextConf()
            if ocrResult:
                text[bbox] = (ocrResult, conf)
                if display:
                    print (u"Box[{0}]: x={3}, y={4}, w={5}, h={6}, "
                           "confidence: {1}, text: {2}").format(i, conf, ocrResult, *bbox)
    return text


def get_best_template_match(img, template, tol):
    h, w = template.shape
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    min_val, _, top_left, _ = cv2.minMaxLoc(res)
    bottom_right = (top_left[0] + w, top_left[1] + h)
    if min_val <= tol:
        return top_left, bottom_right
    else:
        return (None, None)

    
def get_url_box(img, y_offset=-120, url_identifier='https'):
    if y_offset < 0:
        y_offset += img.shape[0]
    bbox = 0, y_offset, img.shape[1], img.shape[0] - y_offset
    for key, (text, conf) in get_segment_text(img, [bbox]).items():
        if url_identifier in text:
            return key, text
    return None, ''
