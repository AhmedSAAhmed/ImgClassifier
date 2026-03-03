import argparse
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
import json
from collections import OrderedDict

def get_input_args():
    '''command-line arguments for prediction '''
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping class to flower name')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    ''' Load the trained model checkpoint '''
    checkpoint = torch.load(checkpoint_path)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    ''' Process an image for use in a PyTorch model '''
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    if width < height:
        new_width = 256
        new_height = int(height * 256 / width)
    else:
        new_height = 256
        new_width = int(width * 256 / height)
    pil_image = pil_image.resize((new_width, new_height))

    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    pil_image = pil_image.crop((left, top, right, bottom))

    np_image = np.array(pil_image) / 255.0
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, top_k=5, device="cpu"):
    ''' Predict the top K classes for an image '''
    model.to(device)
    model.eval()

    np_image = process_image(image_path)
    image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logps = model(image_tensor)
        ps = torch.exp(logps)
        top_probs, top_indices = ps.topk(top_k, dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[int(idx)] for idx in top_indices[0]]

    return top_probs[0].cpu().numpy(), top_classes

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint)
    top_probs, top_classes = predict(args.image_path, model, args.top_k, device)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_flowers = [cat_to_name[class_id] for class_id in top_classes]
    else:
        top_flowers = top_classes

    print("Top Predictions:")
    for i in range(args.top_k):
        print(f"{top_flowers[i]}: {top_probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()
