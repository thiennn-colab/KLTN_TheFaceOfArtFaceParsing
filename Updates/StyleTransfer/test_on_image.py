from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from torchvision.utils import save_image
from PIL import Image

def exportStyleTransfer(image_path, style):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    styles = ['9', '32', '51', '55', '56', '58', '108', '140', '150', '153', '154', '155', '156']
    checkpoint_model = '/home/KLTN_TheFaceOfArtFaceParsing/Updates/StyleTransfer/models/' + styles[style] + '_4000.pth'
    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(checkpoint_model))
    transformer.eval()

    # Prepare input
    image_tensor = Variable(transform(Image.open(image_path))).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()

    # Save image
    save_image(stylized_image,'/home/KLTN_TheFaceOfArtFaceParsing/result.jpg')