import random
import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from PIL import Image

def makeVideoFromImageArray(output_filename, image_list):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Defines the codec to be used for the video. XVID is a popular video codec.

    image_width = image_list[0].shape[1]
    image_height = image_list[0].shape[0]
    # Retrieves the width and height of the images from the first image in the list.
    # It assumes all images are of the same size.

    out = cv2.VideoWriter(filename=output_filename, fourcc=fourcc, fps=8,
                          frameSize=(image_width, image_height), isColor=True)
    # loop over the list of pictures
    for image_number, image in enumerate(image_list, 1):
        # Create a text overlay with the image number
        text = f"Frame {image_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX # Chooses the font FONT_HERSHEY_SIMPLEX.
        font_scale = 0.5
        font_thickness = 1
        # Sets the font scale to 0.5 and thickness to 1.
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = 10
        text_y = 20

        # Add the text to the frame
        frame_with_text = image.copy()

        # Add the text to the frame
        # Uses cv2.putText to overlay the text on the image.
        cv2.putText(frame_with_text, text, (text_x, text_y), font,
                    font_scale, (255, 255, 255), font_thickness)

        out.write(frame_with_text)

    out.release()

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".bmp") or filename.endswith(".png"):
            # Create the full file path
            file_path = os.path.join(folder_path, filename)
            # Read the image using OpenCV
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image: {filename}")
    return images


def load_images_gn(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".bmp") or filename.endswith(".png"):
            # Create the full file path
            file_path = os.path.join(folder_path, filename)
            # Read the image using OpenCV
            # mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Reads the mask in grayscale
            mask = Image.open(file_path).convert('L')
            mask_np = np.array(mask)
            binarized_mask = ((mask_np > 0).astype(np.uint8)).astype(np.uint8)
            if mask is not None:
                images.append(binarized_mask)
            else:
                print(f"Failed to load image: {filename}")
    return images

def load_json_annotations(json_path):
    with open(json_path, 'r') as json_file:
        annotations = json.load(json_file)
    return annotations


def nullify_outside_boxes(width, height, bounding_boxes):
    """
    Nullify activations outside bounding boxes.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        bounding_boxes (list of dict): List of bounding boxes with keys 'x1', 'y1', 'width', and 'height'.

    Returns:
        mask (np.ndarray): Mask with activations outside bounding boxes set to zero.
    """
    # Create a mask initialized to zero
    mask = np.zeros((height, width), dtype=np.uint8)

    for box in bounding_boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = x1 + box[2]
        y2 = y1 + box[3]

        # Ensure bounding box is within the image boundaries
        xx_l = max(x1, 0)
        yy_l = max(y1, 0)
        xx_h = min(x2, width)
        yy_h = min(y2, height)

        # Set mask to 1 inside the bounding box
        mask[yy_l:yy_h, xx_l:xx_h] = 255

    return mask


def inpaint_image(config_Dict,img):

    lst_lower = config_Dict["lower_threshold"]
    lst_upper = config_Dict["upper_threshold"]
    lower = tuple(lst_lower)
    upper = tuple(lst_upper)
    thresh = cv2.inRange(img, lower, upper)

    # Apply morphology operations to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config_Dict["kernel_close"], config_Dict["kernel_close"]))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config_Dict["kernel_dilate"], config_Dict["kernel_dilate"]))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=1)

    hh, ww, _ = img.shape
    black = np.zeros([hh + 2, ww + 2], np.uint8)
    mask = morph.copy()
    mask = cv2.floodFill(mask, black, (0, 0), 0, 0, 0, flags=8)[1] #Mask has same size as original image
    return mask

def convert_to_two_points(bbox):
    """
    Convert bounding box coordinates from (x, y, w, h) format to two points (x1, y1) and (x2, y2).
    """
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1), (x2, y2)


def get_image_foreground_mog2(model,config_Dict, frame, yolo_predictions):

    learning_rate = config_Dict["learning_rate"]
    background_model= model
    fg = background_model.apply(frame, learningRate=learning_rate)

    _, thresh = cv2.threshold(fg, 100, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None)
    thresh = cv2.dilate(thresh, None)

    for box in yolo_predictions:
        x, y, w, h = box
        h_margin = int(h * 2 / 3)
        w_margin = int(w / 5)
        x = x - w_margin
        y = y - h_margin
        w = w + 2 * w_margin
        h = h + 2 * h_margin
        bbox = (x, y, w, h)
        point1, point2 = convert_to_two_points(bbox)
        cv2.rectangle(thresh, point1, point2, 0, -1)

    return thresh
def mask_noise_foreground(mog_mask,inpainted_mask):

    result_image = mog_mask.copy()

    white_white = np.logical_and(mog_mask == 255, inpainted_mask == 255)

    white_black = np.logical_and(mog_mask == 255, inpainted_mask == 0)

    black_white = np.logical_and(mog_mask == 0, inpainted_mask == 255)

    result_image[white_white] = 0

    result_image[white_black] = 255

    result_image[black_white] = 0

    return result_image
def remove_small_noise_by_contour_area(config_Dict, foreground_mask):

    max_contour_area = config_Dict["max_contour_area"]
    min_contour_area = config_Dict["min_contour_area"]

    if max_contour_area is None:
        max_contour_area = 1e6  # Set a large value if max_contour_area is not specified

    image=foreground_mask # this mask is already in grey scal

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(image)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if min_contour_area <= contour_area <= max_contour_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    image_without_small_contours = cv2.bitwise_and(image, image, mask=mask)

    return image_without_small_contours


def highlight_mask_in_image(frame, final_mask):

    b, g, red_channel = cv2.split(frame)

    combined_images = []

    if final_mask.shape[:2] != red_channel.shape[:2]:
        raise ValueError("Binary mask and image must have the same dimensions.")

    red_channel = cv2.add(final_mask, red_channel)

    red_channel = np.clip(red_channel, 0, 255).astype(np.uint8)

    rgb_image = cv2.merge([b, g, red_channel])

    return rgb_image


def detect_foreign_particles(bg_model,config_Dict,frame,yolo_boxes,nullify_mask,show_inpaint_effect=False):
    inpainted_noise = inpaint_image(config_Dict, frame)
    mog_foreground = get_image_foreground_mog2(bg_model,config_Dict, frame, yolo_boxes)


    mask_without_noise = mask_noise_foreground(mog_foreground, inpainted_noise)
    # you have removed the noise
    # params is ForeignParticleDetectorParams inside of it background model
    final_mask = remove_small_noise_by_contour_area(config_Dict, mask_without_noise)
    final_mask = cv2.bitwise_and(final_mask, nullify_mask)
    ##
    height, width, _ = frame.shape
    ##
    line_position1 = int(0.10 * width)
    line_position2 = int(0.90 * width)
    ##
    final_mask[:, :line_position1] = 0
    final_mask[:, line_position2:] = 0
    ##

    # Visualize foreign particle on the original frame
    visualization_activation_map = highlight_mask_in_image(frame, final_mask)

    # Draw the vertical lines
    color = (0, 255, 0)  # Color of the line (B, G, R)
    thickness = 2  # Thickness of the line

    # Draw the first line
    cv2.line(visualization_activation_map, (line_position1, 0), (line_position1, height), color, thickness)

    # Draw the second line
    cv2.line(visualization_activation_map, (line_position2, 0), (line_position2, height), color, thickness)
    if not show_inpaint_effect:
        return visualization_activation_map,final_mask
    else:
        mog_foreground = cv2.bitwise_and(mog_foreground, nullify_mask)
        mog_foreground[:, :line_position1] = 0
        mog_foreground[:, line_position2:] = 0
        visualization_inpaint_effect = highlight_mask_in_image(frame, mog_foreground)
        cv2.line(visualization_inpaint_effect, (line_position1, 0), (line_position1, height), color, thickness)

        # Draw the second line
        cv2.line(visualization_inpaint_effect, (line_position2, 0), (line_position2, height), color, thickness)
        # Stack the images horizontally
        stacked_image = np.hstack((visualization_activation_map, visualization_inpaint_effect))
        return stacked_image,mog_foreground


def reduce_frame_size(frame, yolo_boxes, scale_percent=0.9):
    if scale_percent >= 0.9:
        return frame, yolo_boxes
    else:
        # Resize the frame
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        # Resize the bounding boxes
        for rect in yolo_boxes:
            rect[0] = int(rect[0] * scale_percent)
            rect[1] = int(rect[1] * scale_percent)
            rect[2] = int(rect[2] * scale_percent)
            rect[3] = int(rect[3] * scale_percent)
    return frame, yolo_boxes


def reduce_frame_size_X_Y(frame, yolo_boxes, scale_percent_X, scale_percent_Y):
    if scale_percent_X >= 0.9:
        return frame, yolo_boxes
    else:
        # Resize the frame
        width = 224
        height = 224
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        # Resize the bounding boxes
        for rect in yolo_boxes:
            rect[0] = int(rect[0] * scale_percent_X)
            rect[1] = int(rect[1] * scale_percent_Y)
            rect[2] = int(rect[2] * scale_percent_X)
            rect[3] = int(rect[3] * scale_percent_Y)
    return frame, yolo_boxes

def reduce_gn(frame, scale_percent=0.9):
    if scale_percent >= 0.9:
        return frame
    else:
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def reduce_gn_X_Y(frame, scale_percent_x,scale_percent_y):
    if scale_percent_x >= 0.9:
        return frame
    else:
        width = 224
        height = 224
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def reshape_lanes(lanes,scale_percent):
    for rect in lanes:
        rect[0] = int(rect[0] * scale_percent)
        rect[1] = int(rect[1] * scale_percent)
        rect[2] = int(rect[2] * scale_percent)
        rect[3] = int(rect[3] * scale_percent)
    return lanes


def reshape_lanes_X_Y(lanes,scale_percent_x,scale_percent_y):
    for rect in lanes:
        rect[0] = int(rect[0] * scale_percent_x)
        rect[1] = int(rect[1] * scale_percent_y)
        rect[2] = int(rect[2] * scale_percent_x)
        rect[3] = int(rect[3] * scale_percent_y)
    return lanes


def calculate_sensitivity(tp, fn):
    """
    Calculate sensitivity/ recall (True Positive Rate or Recall).
    """
    return (tp / (tp + fn))*100

# Precision
def calculate_precision(tp, fp):
    """
    Calculate Precision (True Positive Rate or Sensitivity).
    """
    return (tp / (tp + fp))*100


def calculate_accuracy(tp, tn, fp, fn):
    """
    Calculate accuracy.
    """
    correct_predictions = tp + tn
    total_predictions = tp + tn + fp + fn
    return (correct_predictions / total_predictions)*100
if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    config_Dict = {"lower_threshold": (120,120,120),
                   "upper_threshold": (240, 240, 240),
                   "detect_shadows": False,
                   "varThreshold": 16,
                   "learning_rate": -1,
                   "history": 50,
                   "min_contour_area": 500,
                   "max_contour_area": 1000000,
                   "kernel_close": 7,
                   "kernel_dilate": 25,
                   "lane_1_coord": [0, 140,1216,103],
                   "lane_2_coord": [0, 453,1216,110],
                   "lane_3_coord": [0,783,1216,105],
                   "lane_4_coord": [0,1103,1216,125],
                   "lane_5_coord": [0,1438,1216,128],
                   "lane_6_coord": [0,1768,1216,105],
                   "percImageReduction":0.4,
                   "output_path": r"testing.avi",
                   "input_path":
                       r"D:\github_directories\foriegn\black_plate_c_shape_blue_pills\SEQ00004_MIXED_FOREIGN_PARTICLE",
                   "json_path":
                       r"D:\github_directories\foriegn\black_plate_c_shape_blue_pills\SEQ00004_MIXED_FOREIGN_PARTICLE\Labeling.json",
                   }
    folder_path_frames = config_Dict["input_path"]
    gn_path_frames = str(config_Dict["input_path"])+"_mask"
    json_file_path = config_Dict["json_path"]

    foreign_frames = load_images(folder_path_frames)
    yolo_predictions = load_json_annotations(json_file_path)
    foreign_frames_gn = load_images_gn(gn_path_frames)
    localizations_dict = {}
    for item in yolo_predictions['files']:
        file_name = item['file_name']
        localizations = item['annotations']
        if file_name not in localizations_dict:
            localizations_dict[file_name] = []
        bbox_values = [localization["bbox"] for localization in localizations]
        localizations_dict[file_name].extend(bbox_values)
    bbox_lists = list(localizations_dict.values())
    # Calculate scale factors
    height, width = foreign_frames[0].shape[0:2]
    x_scale = 224 / width
    y_scale = 224 / height
    output_sequence = []
    if config_Dict["percImageReduction"] >= 0.9:
        config_Dict["percImageReduction"] = 1.0

    print(height,width)
    # Calculate the new dimensions (50% of the original)
    width = int(width * config_Dict["percImageReduction"])
    height = int(height * config_Dict["percImageReduction"])
    lanes = [config_Dict["lane_1_coord"],
             config_Dict["lane_2_coord"],
             config_Dict["lane_3_coord"],
             config_Dict["lane_4_coord"],
             config_Dict["lane_5_coord"],
             config_Dict["lane_6_coord"]]
    if config_Dict["percImageReduction"] < 0.9:
        # lanes = reshape_lanes(lanes,config_Dict["percImageReduction"])
        lanes = reshape_lanes_X_Y(lanes,x_scale,y_scale)

        # config_Dict["percImageReduction"] = 1.0
    nullify_mask = nullify_outside_boxes(224, 224, lanes)

    # nullify_mask = nullify_outside_boxes(width,height,lanes)
    bg_model = cv2.createBackgroundSubtractorMOG2(history=config_Dict["history"],
                                                  varThreshold=config_Dict["varThreshold"],
                                                  detectShadows=config_Dict["detect_shadows"])
    list_activations = []
    for frame_idx in tqdm(range(len(foreign_frames))):
        # frame, yolo_boxes = reduce_frame_size(foreign_frames[frame_idx],bbox_lists[frame_idx],
        #                                       config_Dict["percImageReduction"])
        frame, yolo_boxes = reduce_frame_size_X_Y(foreign_frames[frame_idx],bbox_lists[frame_idx],x_scale,y_scale)
        # foreign_frames_gn[frame_idx] = reduce_gn(foreign_frames_gn[frame_idx], config_Dict["percImageReduction"])
        foreign_frames_gn[frame_idx] = reduce_gn_X_Y(foreign_frames_gn[frame_idx] , x_scale , y_scale)


        output_image,activation = detect_foreign_particles(bg_model,config_Dict, frame,
                                                           yolo_boxes,nullify_mask,
                                                           show_inpaint_effect=True)
        output_sequence.append(output_image)
        activation = (activation.astype(np.float32)/255.0).astype(np.uint8)
        activation = np.expand_dims(activation, axis=0)
        list_activations.append(activation)
    makeVideoFromImageArray(config_Dict["output_path"], output_sequence)

    seg_scores = np.asarray(list_activations)
    gt_mask = (np.asarray(foreign_frames_gn)).astype(np.uint8)
    gt_flat = gt_mask.flatten()
    pred_flat = seg_scores.flatten()
    print("Ground Truth shape",gt_flat.shape)
    print("Predicted shape",pred_flat.shape)

    per_pixel_rocauc = roc_auc_score(gt_flat, pred_flat)
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(gt_flat, pred_flat)
    goal_matrix = confusion_matrix(gt_flat, gt_flat)
    print("Target Goal")
    print(goal_matrix)
    print("algo performance")
    print(conf_matrix)
    # True positives (TP) are in the intersection of positive class in both gt and predictions
    true_positives = conf_matrix[1, 1]
    true_negatives = conf_matrix[0, 0]

    # False positives (FP) are positive in predictions but negative in gt
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]

    perc_matrix = np.array([[(conf_matrix[0, 0]/goal_matrix[0,0])*100,
                            -1],
                           [-1,
                            (conf_matrix[1, 1]/goal_matrix[1,1])*100]])

    print("perc_matrix")
    print(perc_matrix)
    # Average number of true positive activations per pixel
    average_true_positive = true_positives / np.sum(gt_flat == 1)

    # Average number of false positive activations per pixel
    # we divide the false postive by the total number no activations
    average_false_positive = false_positives / np.sum(gt_flat == 0)
    print('average true postive: {0}'.format(average_true_positive))
    print('average false postive: {0}'.format(average_false_positive))

    print("The Sensetivity/Recall is: {0}".format(calculate_sensitivity(true_positives, false_negatives)))
    print("The Precsion is: {0}".format(calculate_precision(true_positives, false_positives)))
    print("The accuracy is: {0}".format(calculate_accuracy(true_positives,
                                                           true_negatives,
                                                           false_positives,
                                                           false_negatives)))
    print('pixel ROCAUC: %.2f' % per_pixel_rocauc)

