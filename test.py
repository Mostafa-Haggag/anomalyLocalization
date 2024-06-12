
from network import VAE_new
import torch
import random
from dataset import return_MVTecAD_loader
from tqdm import tqdm
import numpy as np
import cv2
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




def plot_bounding_boxes(original, reconstructed, threshold=0.5):
    # Compute the absolute difference

    difference = np.abs(original - reconstructed)

    # Threshold the difference
    difference_gray = np.max(difference, axis=2)
    mask = (difference_gray > threshold).astype(np.uint8)

    # Convert mask to uint8
    mask = (mask * 255).astype(np.uint8)
    if mask is None or mask.size == 0:
        raise ValueError("The mask image is empty or not valid.")

    # Ensure the mask is in the correct format (binary image)
    if len(mask.shape) != 2 or mask.dtype != np.uint8:
        raise ValueError("The mask image must be a binary image of type np.uint8.")

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on the original image
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Filter small contours if needed
            print("Contour detected")
            x, y, w, h = cv2.boundingRect(contour)
            try:
                cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
            except cv2.error as e:
                print(x,y,w,h)
    return original


if __name__ == '__main__':
    seed =1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    config_dict = {"batch_size": 1,
                   "epoch": 500,
                   "lr": 5e-4,
                   "z_dim": 512,
                   "model_id":'vae_reconstruction',
                   "tag":"long_run"
                   }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE_new(z_dim=config_dict["z_dim"]).to(device)
    path_model = \
        r"D:\github_directories\anomalyLocalization\checkpoints\20240610_142651_vae_reconstruction_long_run\30.pth"
    model.load_state_dict(torch.load(path_model), strict=True)
    test_loader = return_MVTecAD_loader(image_dir=r"D:\github_directories\foriegn\SEQ00004_MIXED_FOREIGN_PARTICLE",
                                         batch_size=config_dict["batch_size"], train=False)
    model.eval()
    output_sequence = []
    output_sequence_reconstruced = []

    for x in tqdm(test_loader):
        with torch.no_grad():
            x= x.to(device)
            x_hat = model(x)
            x = x.squeeze(0).detach().cpu().numpy()
            x_output = x_hat.squeeze(0).detach().cpu().numpy()
            # original = (x - x.min()) / (x.max() - x.min())
            original = x
            # reconstructed = (x_output - x_output.min()) / (x_output.max() - x_output.min())
            reconstructed = x_output
            new_frame = plot_bounding_boxes(original=original,reconstructed=reconstructed,threshold=0.1)
            new_frame_2 = cv2.cvtColor(np.clip(new_frame.transpose(1, 2, 0)*255,0,255).astype(np.uint8),
                                       cv2.COLOR_RGB2BGR)
            new_frame_reconstruced = cv2.cvtColor(np.clip(reconstructed.transpose(1, 2, 0)*255,0,255).astype(np.uint8),
                                       cv2.COLOR_RGB2BGR)
            output_sequence_reconstruced.append(new_frame_reconstruced)
            output_sequence.append(new_frame_2)

    makeVideoFromImageArray('video.avi', output_sequence)
    makeVideoFromImageArray('video_constructed.avi', output_sequence_reconstruced)
