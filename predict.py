import time
import torch
import json
import numpy as np
import argparse, sys
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                   }

# TODO: Load the datasets with ImageFolder
directories = {'train': train_dir, 
               'valid': valid_dir, 
               'test' : test_dir}
image_datasets = {x: datasets.ImageFolder(directories[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}
# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}


# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_model(checkpoint_path):
    
    check_path = torch.load(checkpoint_path)
    # add for test
    arch = "vgg13"
    
    if (arch == 'vgg13'):
        model = models.vgg13(pretrained=True)
        input_size = 25088
        hidden_units = 4096
        output_size = 102
    elif (arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
        input_size = 1024
        hidden_units = 500
        output_size = 102
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = check_path['class_to_idx']
    
    # Build a feed-forward network
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(4096, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(check_path['state_dict'])
    
    return model

model = loading_model('save_checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    # Edit
    edit_image = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Dimension
    img_tensor = edit_image(pil_image)
    processed_image = np.array(img_tensor)
    processed_image = processed_image.transpose((0, 2, 1))
    
    
    return processed_image


# Test image after process
image_path = 'flowers/test/1/image_06743.jpg'
img = process_image(image_path)
print(img.shape) 


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
print("done")
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    # Process image
    image = process_image(image_path)
    # Transfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)
    output = model.forward(image)
    # Top probs
    probabilities = torch.exp(output).data
    top_probs, top_labs = probabilities.topk(topk)
    top_probs = top_probs.cpu().detach().numpy().tolist()[0]
    top_labs = top_labs.cpu().detach().numpy().tolist()[0]
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    return top_probs, top_labels
print("done")
# TODO: Display an image along with the top 5 classes
# TODO: Display an image along with the top 5 classes
def display_top5(image_path, model):
    
    # Setting plot area
    plt.figure(figsize = (3,6))
    ax = plt.subplot(2,1,1)
    
    # Display test flower
    img = process_image(image_path)
    get_title  = image_path.split('/')
    print(cat_to_name[get_title[2]])
    imshow(img, ax, title = cat_to_name[get_title[2]]);
    
    # Making prediction
    score, flowers_list = predict(image_path, model) 
    fig,ax = plt.subplots(figsize=(4,3))
    sticks = np.arange(len(flowers_list))
    ax.barh(sticks, score, height=0.3, linewidth=2.0, align = 'center')
    ax.set_yticks(ticks = sticks)
    ax.set_yticklabels(flowers_list)
    return image_path, model
image_path = 'flowers/test/100/image_07896.jpg'
get_title  = image_path.split('/')
print("Test image:" + cat_to_name[get_title[2]])
save_result_dir = 'save_prediction_result_1'
display_top5(image_path, model)
print("Save prediction result to:" + save_result_dir)
print("Prediction result:")
score, flower_list = predict(image_path, model)
print(flower_list)
print(np.exp(score))
print("-------------------------------------------")


