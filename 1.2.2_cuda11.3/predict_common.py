import cv2
import io
import numpy as np

from typing import List
from PIL import Image


PREDICTION_FORMAT_GRAYSCALE = "grayscale"
PREDICTION_FORMAT_BLUECHANNEL = "bluechannel"
PREDICTION_FORMAT_INDEXED = "indexed"
PREDICTION_FORMATS = [
    PREDICTION_FORMAT_GRAYSCALE,
    PREDICTION_FORMAT_BLUECHANNEL,
    PREDICTION_FORMAT_INDEXED,
]


def create_palette(num_colors: int) -> List[int]:
    """
    Returns a list of palette entries (R,G,B) with the specified number of colors.

    :param num_colors: the number of colors to generate
    :type num_colors: int
    :return: the generated list of colors
    :rtype: list
    """
    return [1 + i // 3 for i in range(3*num_colors)]


def fill_palette(palette: List[int]) -> List[int]:
    """
    Makes sure that there are 256 R,G,B values present. Simply adds grayscale R,G,B values.

    :param palette: the palette to fill up, if necessary
    :type palette: list
    :return: the (potentially) updated list of R,G,B values
    :rtype: list
    """
    if len(palette) < 256*3:
        if len(palette) % 3 != 0:
            raise ValueError("Palette does not contain multiples of three (ie R,G,B values)!")
        palette = palette + create_palette(256 - (len(palette) // 3))
    return palette


def default_palette() -> List[int]:
    """
    Returns a palette of 255 R,G,B triplets all in a single list, to be used in indexed PNG files.
    Black is always the first color.

    :return: the flat list of R,G,B values
    :rtype: list
    """
    result = [0, 0, 0,      # black
              255, 0, 0,    # red
              0, 255, 0,    # green
              0, 0, 255,    # blue
              255, 0, 255,  # magenta
              255, 255, 0,  # yellow
              0, 255, 255]  # cyan
    return result


def prediction_to_file(prediction, prediction_format: str, path: str):
    """
    Saves the mask prediction to disk as image using the specified image format.

    :param prediction: the mmsegmentation prediction object
    :param prediction_format: the image format to use
    :type prediction_format: str
    :param path: the path to save the image to
    :type path: str
    """
    if prediction_format not in PREDICTION_FORMATS:
        raise Exception("Unsupported format: %s" % prediction_format)

    pr_mask = prediction.pred_sem_seg
    pr_mask = np.array(pr_mask.cpu().values()[0], dtype=np.uint8)
    pr_mask = np.transpose(pr_mask, (1, 2, 0))

    if prediction_format == PREDICTION_FORMAT_GRAYSCALE:
        cv2.imwrite(path, pr_mask)
    elif prediction_format == PREDICTION_FORMAT_BLUECHANNEL:
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)
        pr_mask[:, :, 1] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        pr_mask[:, :, 2] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        cv2.imwrite(path, pr_mask)
    elif prediction_format == PREDICTION_FORMAT_INDEXED:
        pr_mask = np.squeeze(pr_mask)
        pr_pil = Image.fromarray(pr_mask, "L")
        pr_pil.putpalette(default_palette())
        pr_pil.save(path)
    else:
        raise Exception("Unhandled format: %s" % prediction_format)


def prediction_to_bytes(prediction, prediction_format: str) -> bytes:
    """
    Turns the mask prediction into bytes using the specified image format.

    :param prediction: the mmsegmentation prediction object
    :param prediction_format: the image format to use
    :type prediction_format: str
    :return: the generated image
    :rtype: bytes
    """
    if prediction_format not in PREDICTION_FORMATS:
        raise Exception("Unsupported format: %s" % prediction_format)

    pr_mask = prediction.pred_sem_seg
    pr_mask = np.array(pr_mask.cpu().values()[0], dtype=np.uint8)
    pr_mask = np.transpose(pr_mask, (1, 2, 0))

    if prediction_format == PREDICTION_FORMAT_GRAYSCALE:
        result = cv2.imencode('.png', pr_mask)[1].tobytes()
    elif prediction_format == PREDICTION_FORMAT_BLUECHANNEL:
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)
        pr_mask[:, :, 1] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        pr_mask[:, :, 2] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        result = cv2.imencode('.png', pr_mask)[1].tobytes()
    elif prediction_format == PREDICTION_FORMAT_INDEXED:
        pr_mask = np.squeeze(pr_mask)
        pr_pil = Image.fromarray(pr_mask, "L")
        pr_pil.putpalette(default_palette())
        buffer = io.BytesIO()
        pr_pil.save(buffer, format="PNG")
        result = buffer.getvalue()
    else:
        raise Exception("Unhandled format: %s" % prediction_format)

    return result
