
from network import VAE_new
import torch
import random
from dataset import return_MVTecAD_loader,return_MVTecAD_loader_test_GN
from kornia.filters import gaussian_blur2d

from tqdm import tqdm
import numpy as np
import cv2
from unet import UNetModel,UNetModel_noskipp
from torchvision.transforms import transforms
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pandas as pd

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
        # cv2.putText(frame_with_text, text, (text_x, text_y), font,
        #             font_scale, (255, 255, 255), font_thickness)

        out.write(frame_with_text)

    out.release()

def makeVideoFromImageArray_uncolored(output_filename, image_list):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    image_height, image_width = image_list[0].shape[:2]

    out = cv2.VideoWriter(filename=output_filename, fourcc=fourcc, fps=8,
                          frameSize=(image_width, image_height), isColor=False)

    for image_number, image in enumerate(image_list, 1):
        # # Convert single-channel image to 3-channel (if needed)
        # if len(image.shape) == 2:
        #     # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #     image
        # Add text overlay
        text = f"Frame {image_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = 10
        text_y = 20
        # cv2.putText(image, text, (text_x, text_y), font, font_scale, 255, font_thickness)

        out.write(image)

    out.release()
# def convert_to_grayscale(image):
#     """
#     Convert an image to grayscale.
#     """
#     return cv2.cvtColor( cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
# def convert_to_grayscale(image):
#     """
#     Convert a NumPy array of shape (3, H, W) representing an RGB image to grayscale.
#     """
#     # Ensure the input image is in the format (H, W, 3) for easier manipulation
#     image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, 3)
#
#     # Apply the grayscale conversion weights
#     gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
#
#     return gray_image
def convert_to_grayscale(image):
    """
    Convert an RGB image to grayscale.
    """
    # Ensure the input image is in the format (H, W, 3) for easier manipulation
    image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, 3)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def mse(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred))

def plot_bounding_boxes(original, reconstructed, threshold=0.1):
    """
    Plot bounding boxes around areas with significant differences.
    """
    # difference =mse(original,reconstructed)
    difference = np.abs(original - reconstructed)
    #difference_gray = np.mean(difference, axis=0)
    # difference_gray = np.sum(difference, axis=0)
    difference_gray = np.max(difference, axis=0)
    mask = ((difference_gray > threshold).astype(np.uint8) * 255).astype(np.uint8)



    # Convert original and reconstructed to grayscale
    # original_gray = convert_to_grayscale(original)
    # reconstructed_gray = convert_to_grayscale(reconstructed)

    # Compute the absolute difference
    # difference_gray = np.abs(original_gray - reconstructed_gray)

    # Threshold the difference
    # mask = (difference_gray > threshold ).astype(np.uint8)

    # Ensure the mask is in the correct format (binary image)
    if mask is None or mask.size == 0:
        raise ValueError("The mask image is empty or not valid.")
    if len(mask.shape) != 2 or mask.dtype != np.uint8:
        raise ValueError("The mask image must be a binary image of type np.uint8.")



    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the original image to BGR format for drawing rectangles
    original_bgr = np.transpose(original, (1, 2, 0))  # Convert to (H, W, 3)
    original_bgr = (original_bgr * 255).astype(np.uint8)  # Scale to [0, 255]
    original_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_RGB2BGR)


    red_channel = original_bgr[:, :, 2]

    # Applica la maschera al canale rosso
    red_channel = np.maximum(red_channel, mask)
    # Apply morphological operations to remove small noise and fill gaps
    # kernel = np.ones((2, 2), np.uint8)
    # red_channel = cv2.erode(red_channel, kernel, iterations=1)
    # red_channel = cv2.dilate(red_channel, kernel, iterations=1)
    # Ricombina i canali dell'immagine
    original_bgr[:, :, 2] = red_channel

    # Draw bounding boxes on the original image

    # for contour in contours:
    #     if cv2.contourArea(contour) > 50:  # Filter small contours if needed
    #         #print("Contour detected")
    #         # cv2.drawContours(original_bgr, contour, -1, (0, 255, 0), 3)
    #
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(original_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return original_bgr,mask


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


# def pixel_distance(output, target):
#     '''
#     Pixel distance between image1 and image2
#     '''
#     # Compute absolute difference between output and target images
#     abs_diff = np.abs(output - target)
#
#     # Compute mean along channel dimension
#     mean_diff = np.mean(abs_diff, axis=0)
#
#     # Add channel dimension
#     distance_map = np.expand_dims(mean_diff, axis=0)
#
#     return distance_map
def pixel_distance(output, target):
    '''
    Pixel distance between image1 and image2
    '''
    distance_map = torch.mean(torch.abs(output - target), dim=0).unsqueeze(0).unsqueeze(0)
    return distance_map

if __name__ == '__main__':
    seed =1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    config_Dict = {"batch_size": 1,
                   "epoch": 20,
                   "lr": 5e-4,
                   "z_dim": 512,
                   "model_id": 'VAE_WITH_crazy_design',
                   "tag": "long_run",
                   "unet_inout_channels": 3,
                   "unet_inplanes": 16,
                   "unet_residual": 1,
                   "unet_attention": [8],
                   "unet_dropout": 0.0,
                   "unet_mult": [1,2,3,4],
                   "unet_resample": False,
                   "unet_dims": 2,
                   "unet_checkpoint": False,
                   "unet_heads": 4,
                   "unet_num_heads_channels": 4,
                   "unet_res_updown": False,
                   }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #UNetModel
    model = UNetModel_noskipp(in_channels=config_Dict['unet_inout_channels'],
                      model_channels=config_Dict['unet_inplanes'],
                      out_channels=config_Dict['unet_inout_channels'],
                      num_res_blocks=config_Dict['unet_residual'],
                      attention_resolutions=config_Dict['unet_attention'],
                      dropout=config_Dict['unet_dropout'],
                      channel_mult=config_Dict['unet_mult'],
                      conv_resample=config_Dict['unet_resample'],
                      dims=config_Dict['unet_dims'],
                      use_checkpoint=config_Dict['unet_checkpoint'],
                      num_heads=config_Dict['unet_heads'],
                      num_head_channels =config_Dict['unet_num_heads_channels'],
                      resblock_updown=config_Dict['unet_res_updown'],
                      ).to(device)
    # model = VAE_new(z_dim=config_dict["z_dim"]).to(device)
    path_model = \
        r"D:\github_directories\anomalyLocalization\checkpoints\20240619_153346_UNET_DESIGN_u_net_no_skip_connections\20.pth"
    model.load_state_dict(torch.load(path_model), strict=True)
    test_loader = return_MVTecAD_loader_test_GN(
            image_dir=r"D:\github_directories\foriegn\black_plate_c_shape_blue_pills\SEQ00004_MIXED_FOREIGN_PARTICLE",
            image_size=224,
            batch_size=config_Dict["batch_size"])
    model.eval()
    output_sequence = []
    output_sequence_reconstruced = []
    output_sequence_ = []
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.0622145, 1 / 0.0864737, 1 / 0.07538847]),
                                   transforms.Normalize(mean=[-0.09039213, -0.08525948, -0.09549119],
                                                        std=[1., 1., 1.]),
                                   ])
    MSE = torch.nn.MSELoss()
    gt_mask_list, seg_scores,anomaly_map_list = [],[],[]
    values_mse = []
    values_mask = []
    for index,(x,mask_gn) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            x = x.to(device)
            x_hat = model(x)
            # x = invTrans(x.squeeze(0)).detach().cpu().numpy()
            # x_output = invTrans(x_hat.squeeze(0)).detach().cpu().numpy()
            mse = MSE(x,x_hat)
            values_mse.append(mse.item())
            values_mask.append(torch.max(mask_gn).item())
            x = (x.squeeze(0)).detach().cpu().numpy()
            x_output = (x_hat.squeeze(0)).detach().cpu().numpy()
            # original = (x - x.min()) / (x.max() - x.min())
            original = x
            # reconstructed = (x_output - x_output.min()) / (x_output.max() - x_output.min())
            reconstructed = x_output
            new_frame,mask_calculated = plot_bounding_boxes(original=original,reconstructed=reconstructed,threshold=0.12)
            original_tensor = torch.from_numpy(original)
            reconstructed_tensor = torch.from_numpy(reconstructed)
            # similar to diffusion model
            # i_d = pixel_distance(reconstructed_tensor, reconstructed_tensor)
            # sigma = 4
            # kernel_size = 2 * int(4 * sigma + 0.5) + 1
            # anomaly_map = gaussian_blur2d(i_d, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
            # anomaly_map_list.append(anomaly_map.squeeze(0).detach().cpu().numpy())
            # End to diffusion model
            seg_sig = 6
            mask_calculated = np.expand_dims(mask_calculated, axis=0)
            output_sequence_.append(mask_calculated.transpose(1,2,0))

            # The purpose of applying the Gaussian filter in your code snippet seems to be to smooth the
            # predicted masks before further processing or evaluation.
            # for i in range(mask_calculated.shape[0]):
            #     mask_calculated[i] = gaussian_filter(mask_calculated[i], sigma=seg_sig)

            # mask_calculated = gaussian_filter(mask_calculated, sigma=seg_sig)
            seg_scores.extend(mask_calculated)
            gt_mask_list.extend(mask_gn.squeeze(0).cpu().numpy())

            new_frame_reconstruced = cv2.cvtColor((reconstructed.transpose(1, 2, 0)*255).astype(np.uint8),
                                       cv2.COLOR_RGB2BGR)

            output_sequence_reconstruced.append(new_frame_reconstruced)
            output_sequence.append(new_frame)

    seg_scores = np.asarray(seg_scores).astype(np.float32)
    seg_scores /= 255.0  # Normalize to range [0, 1]
    seg_scores = seg_scores.astype(np.uint8)
    # anomaly_map_list = (np.asarray(anomaly_map_list)).astype(np.uint8)
    # anomaly_map_list = (
    #             (anomaly_map_list - anomaly_map_list.min()) / (anomaly_map_list.max() - anomaly_map_list.min())).astype(np.uint8)

    gt_mask = (np.asarray(gt_mask_list)).astype(np.uint8)

    # Flatten the arrays if needed
    gt_flat = gt_mask.flatten()
    pred_flat = seg_scores.flatten()
    print("Ground Truth shape",gt_flat.shape)
    print("Predicted shape",pred_flat.shape)
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
    print(f"Shape of values_mse: {len(values_mse)}")
    print(f"Shape of values_mask: {len(values_mask)}")
    assert len(values_mse) == len(values_mask), "Mismatched lengths between values_mse and values_mask"

    # df = pd.DataFrame([values_mse,values_mask], columns=['Values','label'])
    df = pd.DataFrame({
        'Values': values_mse,
        'Label': values_mask
    })
    file_excel_name = "meow.xlsx"
    mean_value = df['Values'].mean()
    std_value = df['Values'].std()
    min_value = df['Values'].min()
    max_value = df['Values'].max()
    stats_df = pd.DataFrame({
        'Mean': [mean_value],
        'Std': [std_value],
        'Min': [min_value],
        'Max': [max_value]
    })
    file_path_statistics = "meow_statistics.xlsx"
    df.to_excel(file_excel_name, index=False)
    stats_df.to_excel(file_path_statistics, index=False)

    # Average number of true positive activations per pixel
    average_true_positive = true_positives / np.sum(gt_flat == 1)

    # Average number of false positive activations per pixel
    # we divide the false postive by the total number no activations
    average_false_positive = false_positives / np.sum(gt_flat == 0)
    print('average true postive: {0}'.format(average_true_positive))
    print('average false postive: {0}'.format(average_false_positive))

    print("The Sensetivity/Recall is: {0}".format(calculate_sensitivity(true_positives,false_negatives)))
    print("The Precsion is: {0}".format(calculate_precision(true_positives,false_positives)))
    print("The accuracy is: {0}".format(calculate_accuracy(true_positives,
                                                           true_negatives,
                                                           false_positives,
                                                           false_negatives)))
    # per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), seg_scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), seg_scores.flatten())

    print('pixel ROCAUC: %.2f' % (per_pixel_rocauc))
    makeVideoFromImageArray_uncolored('mask.avi', output_sequence_)
    makeVideoFromImageArray('video.avi', output_sequence)
    makeVideoFromImageArray('video_constructed.avi', output_sequence_reconstruced)
