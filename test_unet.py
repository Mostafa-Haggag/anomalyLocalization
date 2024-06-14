
from network import VAE_new
import torch
import random
from dataset import return_MVTecAD_loader
from tqdm import tqdm
import numpy as np
import cv2
from unet import UNetModel,UNetModel_noskipp
from torchvision.transforms import transforms

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
    difference_gray = np.sum(difference, axis=0)
    #difference_gray = np.max(difference, axis=0)
    mask = ((difference_gray > threshold) * 255).astype(np.uint8)



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
        r"D:\github_directories\anomalyLocalization\checkpoints\20240613_130503_UNET_DESIGN_no_skip\20.pth"
    model.load_state_dict(torch.load(path_model), strict=True)
    test_loader = return_MVTecAD_loader(image_dir=r"D:\github_directories\foriegn\SEQ00004_MIXED_FOREIGN_PARTICLE",
                                         batch_size=config_Dict["batch_size"], train=False)
    model.eval()
    output_sequence = []
    output_sequence_reconstruced = []
    output_sequence_ = []
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.0622145, 1 / 0.0864737, 1 / 0.07538847]),
                                   transforms.Normalize(mean=[-0.09039213, -0.08525948, -0.09549119],
                                                        std=[1., 1., 1.]),
                                   ])
    for x in tqdm(test_loader):
        with torch.no_grad():
            x= x.to(device)
            x_hat = model(x)
            # x = invTrans(x.squeeze(0)).detach().cpu().numpy()
            # x_output = invTrans(x_hat.squeeze(0)).detach().cpu().numpy()
            x = (x.squeeze(0)).detach().cpu().numpy()
            x_output = (x_hat.squeeze(0)).detach().cpu().numpy()
            # original = (x - x.min()) / (x.max() - x.min())
            original = x
            # reconstructed = (x_output - x_output.min()) / (x_output.max() - x_output.min())
            reconstructed = x_output
            new_frame,mask = plot_bounding_boxes(original=original,reconstructed=reconstructed,threshold=0.21)
            new_frame_reconstruced = cv2.cvtColor((reconstructed.transpose(1, 2, 0)*255).astype(np.uint8),
                                       cv2.COLOR_RGB2BGR)
            output_sequence_reconstruced.append(new_frame_reconstruced)
            output_sequence.append(new_frame)
            output_sequence_.append(mask)
    makeVideoFromImageArray('mask.avi', output_sequence_)
    makeVideoFromImageArray('video.avi', output_sequence)
    makeVideoFromImageArray('video_constructed.avi', output_sequence_reconstruced)
