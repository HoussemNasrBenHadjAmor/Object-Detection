import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def save_box_loss(loss, OUT_DIR, title):
    # Check if the directory exists, if not, create it
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)

    
def save_cls_loss(loss, OUT_DIR, title):
    # Check if the directory exists, if not, create it
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)
        
    figure = plt.figure(figsize=(10, 5))
    train_ax = figure.add_subplot(1, 1, 1)
    train_ax.plot(loss, label='Train Loss')
    train_ax.set_title(title)
    train_ax.set_xlabel('Epochs')
    train_ax.set_ylabel('Loss')
    train_ax.legend()
    figure.savefig(f"{OUT_DIR}/{title}.png")
    plt.close(figure)