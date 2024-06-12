import cv2
import io
import numpy as np
import os

from datetime import datetime
from typing import List, Union, Dict
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from PIL import Image
from predict_masks import mask_to_polygon, polygon_to_lists

ENV_CLASSES = "MMSEG_CLASSES"

PREDICTION_FORMAT_GRAYSCALE = "grayscale"
PREDICTION_FORMAT_BLUECHANNEL = "bluechannel"
PREDICTION_FORMAT_INDEXED = "indexed"
PREDICTION_FORMAT_OPEX = "opex"
PREDICTION_FORMATS = [
    PREDICTION_FORMAT_GRAYSCALE,
    PREDICTION_FORMAT_BLUECHANNEL,
    PREDICTION_FORMAT_INDEXED,
    PREDICTION_FORMAT_OPEX,
]


def create_palette(num_colors: int) -> List[int]:
    """
    Returns a list of palette entries (R,G,B) with the specified number of colors.

    :param num_colors: the number of colors to generate
    :type num_colors: int
    :return: the generated list of colors
    :rtype: list
    """
    return [1 + i // 3 for i in range(3 * num_colors)]


def fill_palette(palette: List[int]) -> List[int]:
    """
    Makes sure that there are 256 R,G,B values present. Simply adds grayscale R,G,B values.

    :param palette: the palette to fill up, if necessary
    :type palette: list
    :return: the (potentially) updated list of R,G,B values
    :rtype: list
    """
    if len(palette) < 256 * 3:
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
    result = [0, 0, 0,  # black
              255, 0, 0,  # red
              0, 255, 0,  # green
              0, 0, 255,  # blue
              255, 0, 255,  # magenta
              255, 255, 0,  # yellow
              0, 255, 255]  # cyan
    return result


def mask_to_opex(pr_mask, id_: str, ts: str, mask_nth: int = 1, classes: Dict[int,str] = None) -> ObjectPredictions:
    """
    Turns the segmentation mask into OPEX predictions.

    :param pr_mask: the mask to convert
    :param id_: the ID to use for the predictions
    :type id_: str
    :param ts: the timestamp to use
    :type ts: str
    :param mask_nth: the contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically
    :type mask_nth: int
    :param classes: the index/label relationship dictionary
    :type classes: dict
    :return: the opex predictions
    :rtype: ObjectPredictions
    """
    pr_mask = np.squeeze(pr_mask)
    values = np.unique(pr_mask)
    pred_objs = []
    for value in values:
        if value == 0:
            continue
        sub_mask = np.where(pr_mask == value, pr_mask, 0)
        polys = mask_to_polygon(sub_mask, mask_nth=mask_nth)
        for poly in polys:
            px, py = polygon_to_lists(poly, swap_x_y=True, normalize=False)
            x0 = int(min(px))
            y0 = int(min(py))
            x1 = int(max(px))
            y1 = int(max(py))
            if (x0 < x1) and (y0 < y1):
                bbox = BBox(left=x0, top=y0, right=x1, bottom=y1)
                points = []
                for x, y in zip(px, py):
                    points.append((int(x), int(y)))
                poly = Polygon(points=points)
                label = "object"
                if (classes is not None) and (value in classes):
                    label = classes[value]
                opex_obj = ObjectPrediction(label=label, bbox=bbox, polygon=poly)
                pred_objs.append(opex_obj)
    return ObjectPredictions(id=id_, timestamp=ts, objects=pred_objs)


def prediction_to_file(prediction, prediction_format: str, path: str, mask_nth: int = 1, classes: Dict[int,str] = None) -> str:
    """
    Saves the mask prediction to disk as image using the specified image format.

    :param prediction: the mmsegmentation prediction object
    :param prediction_format: the image format to use
    :type prediction_format: str
    :param path: the path to save the image to
    :type path: str
    :param mask_nth: the contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically
    :type mask_nth: int
    :param classes: the index/label relationship dictionary
    :type classes: dict
    :return: the filename the predictions were saved under
    :rtype: str
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
    elif prediction_format == PREDICTION_FORMAT_OPEX:
        path = os.path.splitext(path)[0] + ".json"
        opex_preds = mask_to_opex(pr_mask, os.path.basename(path), str(datetime.now()), mask_nth=mask_nth, classes=classes)
        opex_preds.save_json_to_file(path)
    else:
        raise Exception("Unhandled format: %s" % prediction_format)

    return path


def prediction_to_data(prediction, prediction_format: str, mask_nth: int = 1, classes: Dict[int,str] = None) -> Union[bytes, str]:
    """
    Turns the mask prediction into bytes using the specified image format.

    :param prediction: the mmsegmentation prediction object
    :param prediction_format: the image format to use
    :type prediction_format: str
    :param mask_nth: the contour tracing can be slow for large masks, by using only every nth row/col, this can be sped up dramatically
    :type mask_nth: int
    :param classes: the index/label relationship dictionary
    :type classes: dict
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
    elif prediction_format == PREDICTION_FORMAT_OPEX:
        ts = str(datetime.now())
        opex_preds = mask_to_opex(pr_mask, ts, ts, mask_nth=mask_nth, classes=classes)
        result = opex_preds.to_json_string()
    else:
        raise Exception("Unhandled format: %s" % prediction_format)

    return result


def classes_dict() -> Dict[int, str]:
    """
    Turns the MMSEG_CLASSES environment variable into a index/label dictionary
    (first label has index=1).

    :return: the generated dictionary
    :rtype: dict
    """
    result = dict()

    if os.getenv(ENV_CLASSES) is not None:
        classes = os.getenv(ENV_CLASSES).split(",")
        for i, cls in enumerate(classes, start=1):
            result[i] = cls

    return result
