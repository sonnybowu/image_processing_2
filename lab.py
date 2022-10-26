#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# VARIOUS FILTERS

#Frame the picture with a black frame of thickness n
def custom_feature(image, n):
    img = {'height': image['height'], 'width': image['width'], 'pixels': image['pixels'][:]}
    height = img['height']
    width = img['width']
    for x in range(height):
        for y in range(width):
            #if its the first row or the left column or the right column
            if x < n or y < n or y > width - n or x > height - n:
                img['pixels'][x * width + y] = 0
    return img
            

#Helper function to get the specified RGB value of a color image
def get_RGB_img_by_color(image, color):
    img = {'height': image['height'], 'width': image['width']}
    if color == 'red':
        img['pixels'] = [pixel[0] for pixel in image['pixels']]
    elif color == 'green':
        img['pixels'] = [pixel[1] for pixel in image['pixels']]
    elif color == 'blue':
        img['pixels'] = [pixel[2] for pixel in image['pixels']]
    else:
        return "No color value given"
    return img

#Combine three grayscale images into one color image arguments must be in the order of red, green, blue
def combine_images(img_red, img_green, img_blue):
    return {'height': img_red['height'], 'width': img_red['width'], 'pixels': [(x, y, z) for x, y, z in zip(img_red['pixels'], img_green['pixels'], img_blue['pixels'])]}

def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color_func(image, func=filt):
        #Get each color from the color image
        red_img = get_RGB_img_by_color(image, "red")
        green_img = get_RGB_img_by_color(image, "green")
        blue_img = get_RGB_img_by_color(image, "blue")
        #Apply the grayscale filter to each RGB img
        func_red_img = func(red_img)
        func_green_img = func(green_img)
        func_blue_img = func(blue_img)
        #Combine the images
        return combine_images(func_red_img, func_green_img, func_blue_img)
    return color_func
        

def make_blur_filter(n):
    def blur_filter(image, n=n):
        return blurred(image, n)
    return blur_filter


def make_sharpen_filter(n):
    def sharpen_filter(image, n=n):
        return sharpened(image, n)
    return sharpen_filter

def make_frame_filter(n):
    def frame_filter(image, n=n):
        return custom_feature(image, n)
    return frame_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def apply_filters(image, filters=filters):
        # We need to convert each filter into its color equivalent
        img = {'height': image['height'], 'width': image['width'], 'pixels': image['pixels'][:]}
        for filter in filters:
            img = filter(img)
        return img
    return apply_filters

# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    im = image
    for i in range(ncols):
        energy_img = compute_energy(greyscale_image_from_color_image(im))
        cumulative_energy = cumulative_energy_map(energy_img)
        min_seam = minimum_energy_seam(cumulative_energy)
        im = image_without_seam(im, min_seam)
    return im


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    img = {'height': image['height'], 'width': image['width']}
    img['pixels'] = [round(.299 * pixel[0] + .587 * pixel[1] + .114 * pixel[2]) for pixel in image['pixels']]
    return img


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    img = {'height': grey['height'], 'width': grey['width'], 'pixels': grey['pixels'][:]}
    return edges(img)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    height = energy['height']
    width = energy['width']
    img = {'height': height, 'width': width, 'pixels': energy['pixels'][:]}
    for x in range(height):
        for y in range(width):
            #If its the first row, don't do anything
            if x != 0:
                l = []
                l.append(img['pixels'][(x - 1) * width + y])
                # Does the top left value exist? If not don't add it to l
                if y - 1 >= 0:
                    l.append(img['pixels'][(x - 1) * width + (y - 1)])
                # Does the top right value exist? If not don't add it to l
                if y + 1 <= width - 1:
                    l.append(img['pixels'][(x - 1) * width + (y + 1)])
                # Add the miniumum value in the list
                img['pixels'][x * width + y] += min(l)
    return img
            



def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    energy_seam = []
    height = cem['height']
    width = cem['width']
    img = {'height': height, 'width': width, 'pixels': cem['pixels'][:]}

    #Find the minumum value on the bottom and its index
    l = []
    for i in range(width):
        l.append(img['pixels'][(height - 1) * width + i])
    min_index = (height - 1) * width + l.index(min(l))
    energy_seam.append(min_index)

    #Get coords of first min value to find adjacent coords
    min_coords = (min_index//width, min_index % width)
    #Get the coords into x and y
    x, y = min_coords[0], min_coords[1]
    #Find the minimum indexes
    for i in reversed(range(height - 1)):
        #l will be a dicstionary that contains the value: coordinates key pair values Note: with dictionary there are no repeated keys which takes care of only taking the leftmost adjacent value
        l = {}
        # Does the top right value exist? If not don't add it to l
        if y + 1 <= width - 1:
            l[img['pixels'][(x - 1) * width + (y + 1)]] = (x - 1, y + 1)

        #Add element that is directly above
        l[img['pixels'][(x - 1) * width + y]] = (x - 1, y)

        # Does the top left value exist? If not don't add it to l
        if y - 1 >= 0:
            l[img['pixels'][(x - 1) * width + (y - 1)]] = (x - 1, y - 1)

        #Get list of the keys/values adjacent 
        keys = list(l.keys())
        #Get the min_val
        min_val = min(keys)
        #Convert the min vals coords into index format
        min_index = (l[min_val][0] * width + l[min_val][1])
        energy_seam.append(min_index)
        #Assign new min coords
        x, y = l[min_val][0], l[min_val][1]
    return energy_seam
    
    
        

def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    img = {'height': image['height'], 'width': image['width'] - 1, 'pixels': image['pixels'][:]}
    for index in seam:
        img['pixels'].pop(index)
    return img


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

# FUNCTIONS FROM LAST LAB
def get_pixel(image, x, y):
    return image['pixels'][image['width'] * x + y]

def set_pixel(image, x, y, c):
    image['pixels'][image['width'] * x + y] = c

def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'][:],
    }        
    for x in range(image['height']):
        for y in range(image['width']):
                color = get_pixel(image, x, y)
                newcolor = func(color)
                set_pixel(result, x, y, newcolor)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c)


# HELPER FUNCTIONS

# This function creates a dictonary of coordinates relative to the origin in key value pairs like (0,1): 1.2
def coords_from_kernel(kernel):
    kernel_layers = math.floor(len(kernel)**(1/2)/2)
    x, y = (0, 0)
    xs = list(range(x - kernel_layers, x + kernel_layers + 1))
    ys = list(range(y - kernel_layers, y + kernel_layers + 1))
    coords = [(x, y) for x in xs for y in ys]
    dict = {}
    for i, coord in enumerate(coords):
        dict[(coord)] = kernel[i]
    return dict

def get_pixel_zero(image, x, y):
    # get height and width of image
    w, h = image['width'], image['height']

    # Check if coordinate is a valid coordinate if it is return color if not return 0
    if (y >= 0 and y < w) and (x >= 0 and x < h):
        return image['pixels'][x*w + y]
    else:
        return 0

def get_pixel_extend(image, x, y):
     # get height and width of image
    w, h = image['width'], image['height']

    # Check if coordinate is a valid coordinate if it is return color if not return 0
    if (y >= 0 and y < w) and (x >= 0 and x < h):
        return image['pixels'][x*w + y]
    #There are only a couple sections where the out of bounds pixel can be. So let's do all the cases for them like in the below picture

    x2, y2 = 0, 0
    #Is the pixel to the left?
    if y < 0:
        #Is the pixel in section 1?
        if x < 0:
            x2 = 0
        #Is the pixel in section 4?
        elif x < h:
            x2 = x
        #if not the first two cases, then it must be in section 6
        else:
            x2 = h - 1
    
    # Is the pixel to the right?
    elif y >= w - 1:
        # Is the pixel in section 3?
        if x < 0:
            y2 = w - 1
        # Is the pixel in section 5?
        elif x < h:
            x2 = x
            y2 = w - 1
        # if not the first two cases, then it must be in section 8
        else:
            x2 = h - 1 
            y2 = w - 1

    # If the pixel is not on the left or right then it must be on top or bottom
    # Is the pixel in section 2?
    elif x < 0:
        y2 = y
    #Only other possibility is if the point is in section 7
    else:
        x2 = h - 1
        y2 = y
    return image['pixels'][x2 * w + y2]

def get_pixel_wrap(image, x, y):
    w, h = image['width'], image['height']
    # Use mod to wrap around for x and y coordinate and return the 1D index
    return image['pixels'][(x % h) * w + y % w]

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings 'zero', 'extend', or 'wrap',
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of 'zero', 'extend', or 'wrap', return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with 'height', 'width', and 'pixels' keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    img = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'][:]
    }
    if boundary_behavior == 'zero':
        for x in range(image['height']):
            for y in range(image['width']):
                new_val = sum([get_pixel_zero(image, x + i, y + j) * kernel[(i, j)] for i, j in kernel.keys()])
                set_pixel(img, x, y, new_val)
        return img

    if boundary_behavior == 'extend':
        for x in range(image['height']):
            for y in range(image['width']):
                new_val = sum([get_pixel_extend(image, x + i, y+ j) * kernel[(i, j)] for i, j in kernel.keys()])
                set_pixel(img, x, y, new_val)
        return img
    
    if boundary_behavior == 'wrap':
        for x in range(image['height']):
            for y in range(image['width']):
                new_val = sum([get_pixel_wrap(image, x +i , y + j) * kernel[(i, j)] for i, j in kernel.keys()])
                set_pixel(img, x, y, new_val)
        return img

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for i, pixel in enumerate(image['pixels']):
        #If the pixel is not an integer then round it
        if not isinstance(pixel, int):
            image['pixels'][i] = round(pixel)
        if pixel > 255:
            image['pixels'][i] = 255
        elif pixel < 0:
            image['pixels'][i] = 0
# FILTERS

#Return a list of vlur kernel
def get_box_blur_kernel(n):
    return [1/(n*n) for i in range(1, n*n + 1)]

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = get_box_blur_kernel(n)
    kernel_coords = coords_from_kernel(kernel)

    # then compute the correlation of the input image with that kernel
    img = correlate(image, kernel_coords, 'extend')

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(img)
    return img

def sharpened(image, n):
    #Get blurred image
    blurred_image = blurred(image, n)
    #scale the original image
    img = {'height': image['height'],
            'width': image['width'],
            'pixels': [2 * pixel for pixel in image['pixels']]}
    #Subtract the blurred image from the scaled image
    img['pixels'] = [sum((x, -y)) for (x, y) in zip(img['pixels'], blurred_image['pixels'])]
    round_and_clip_image(img)
    return img

def edges(image):
    kx = [-1, 0, 1,
            -2, 0, 2,
            -1, 0, 1]
    ky = [-1, -2, -1,
        0,  0,  0,
        1,  2,  1]
    kx_coords = coords_from_kernel(kx)
    ky_coords = coords_from_kernel(ky)
    ox = correlate(image, kx_coords, 'extend')
    oy = correlate(image, ky_coords, 'extend')
    img = {'height': image['height'],
            'width': image['width'],
            'pixels': [round((x**2 + y**2)**(1/2)) for (x, y) in zip(ox['pixels'], oy['pixels'])]}
    round_and_clip_image(img)
    return img

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {"height": h, "width": w, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError("Unsupported image mode: %r" % img.mode)
        w, h = img.size
        return {"height": h, "width": w, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
#     i = {
#     'height': 3,
#     'width': 2,
#     'pixels': [(255, 0, 0), (39, 143, 230), (255, 191, 0),
#                (0, 200, 0), (100, 100, 100), (179, 0, 199)],
# }
    img = load_color_image('test_images/twocats.png')
    gray_img = greyscale_image_from_color_image(img)
    frame = make_frame_filter(10)
    color_frame = color_filter_from_greyscale_filter(frame)
    save_color_image(color_frame(img), 'custom.png')