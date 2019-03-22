from PIL import Image


def load_image_in_PIL(path):
    img = Image.open(path)
    img.load() # Very important for loading large image
    return img