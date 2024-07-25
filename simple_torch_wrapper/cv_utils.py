import PIL

def convert_to_rgba(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts an image to RGBA mode if it has a palette (mode 'P') and contains transparency information.

    Args:
        image (Image.Image): The input image to be converted.

    Returns:
        Image.Image: The converted image in RGBA mode if the original image had a palette and transparency,
                     otherwise returns the original image.
    """
    if image.mode == 'P' and 'transparency' in image.info:
        return image.convert('RGBA')
    return image