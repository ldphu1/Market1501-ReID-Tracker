# Introduction

Here is my python source code for Person Re-Identification (Re-ID) - a robust system for matching human identities across different camera views. With my code, you could:
* Extract discriminative 512-dimensional feature embeddings from human images using a custom ResNet-50 network (`model.py`)
* Train the model using a Triplet-Margin Loss strategy to effectively distinguish different identities (`train.py`)
* Build a feature gallery from a database of known identities (`build_gallery.py`)
* Run an inference app which detects people using YOLOv8 and identifies/tracks them across frames in a single video file (`demo.py`)
  <p align="center">
  <img src="https://github.com/user-attachments/assets/placeholder-image-1" width="30%">
  <img src="https://github.com/user-attachments/assets/placeholder-image-2" width="30%"/>
  <img src="https://github.com/user-attachments/assets/placeholder-image-3" width="30%"/>
  </p>

# Person Re-Identification
In order to use this repo for tracking, you need an input video and a gallery of known identities. When a person appears in the frame, their bounding box will be detected using `yolov8n.pt`. 

The cropped image of the person is then dynamically passed through our trained ResNet-50 model to extract a normalized feature embedding. These embeddings are compared against the pre-built gallery using cosine similarity (via matrix multiplication). If the similarity score is above the threshold (default: `0.6`), the person is assigned their respective ID; otherwise, they are marked as "Unknown".

**Building a Custom Gallery for Specific Videos**

Instead of using the default Market-1501 dataset, you can easily create custom galleries tailored for your specific videos. 

**1. Prepare the Gallery Folder:** Create a new folder (e.g., `my_custom_gallery/`) and place the reference images of the people you want to track inside it.
* **Image Format:** All images must be in `.jpg` format.
  
* **Naming Convention:** The filename **must** start with the Person ID followed by an underscore `_`. The script parses the ID using the string before the first `_`.
  
   *Correct Examples:* `0001_front.jpg`, `0002_camera1.jpg`, `JohnDoe_1.jpg`.
  
   *Incorrect Examples:* `front_0001.jpg`, `image1.png`.
  
**2. Build the Gallery Features:**
Extract features from your custom images by pointing the script to your new folder and specifying a custom save path:
`python3 build_gallery.py --data_dir data/my_custom_gallery/ --save_path weights/my_custom_gallery.pt`.

**3. For video inference:** simply run `python3 demo.py --video_path data/video.avi`. 
  *(Note: You can change the input video path, threshold, and output path inside the arguments of `demo.py`).*

# Dataset

The dataset used for training my model is the **Market-1501** dataset.
The structure requires the standard Market-1501 splits: `bounding_box_train/` for training triplets, and `query/` along with `bounding_box_test/` for evaluation.

# Training

You need to download the **Market-1501** dataset and rename directory to `market1501/`.
1. The training utilizes a custom `triplet_dataset.py` that automatically samples Anchor, Positive, and Negative images for each identity to form training triplets.
2. If you want to train your model with a different set of hyper-parameters, you only need to change the arguments (like `lr`, `margin`, `step_size`) in `train.py`.
3. Then you could simply run PyTorch training using the provided script:
   `python3 train.py --epochs 30 --batch_size 16 --margin 0.3`

# Experiments

The model structure (`model.py`) utilizes a pre-trained **ResNet-50** backbone combined with an Adaptive Average Pooling layer. This is followed by a fully connected layer to reduce the dimensions to 512, and a 1D Batch Normalization layer with L2 normalization for robust feature learning.

I trained the model for 30 epochs using the Adam optimizer and Triplet Margin Loss. The model's performance was monitored using TensorBoard (saved in `weights/`). During training, the model's Rank-1 and Rank-5 accuracies are evaluated on the query set every 5 epochs. 
<p align = "center">
<img width="33%" height="345" alt="image" src="https://github.com/user-attachments/assets/55f9b6e4-adcf-408e-8445-608db24a8a30" />

<img width="66%" height="345" alt="image" src="https://github.com/user-attachments/assets/c5825f81-5372-4739-a26d-e3ce73786443" />
</p>

As shown in the charts above, the loss converges smoothly, and the model achieves impressive final results on the Market-1501 dataset: **Rank-1 accuracy of ~92.3%** and **Rank-5 accuracy of ~97.4%**. The checkpoint with the highest Rank-1 score is automatically saved as (`best_model.pth`).

# Requirements

* python 3.8+
* pytorch
* torchvision
* ultralytics (YOLOv8)
* opencv-python (cv2)
* numpy
* pillow
* tqdm
* tensorboard
