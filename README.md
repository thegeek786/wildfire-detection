**Wildfire Detection**
-A machine learning-based system for detecting wildfires using aerial multispectral image datasets. The system processes paired RGB and thermal images to train a robust detection model for early intervention and disaster management.

**How to Run**

-Step 1: Download the Dataset
-Download the dataset: FLAME-2: Fire Detection and Modeling Aerial Multi-Spectral Image Dataset.
-File: (https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset)
-Extract the dataset to your working directory.
-After extraction, ensure the dataset contains the following folders:
-254p Thermal Images
-254p RGB Images

-Step 2: Setup the Environment
-git clone https://github.com/wildfire-detection.git
-cd wildfire-detection
-Install the required Python packages:
-pip install -r requirements.txt

-Step 3: Preprocess the Dataset
-Run the renaming script to synchronize paired image filenames:
-python rename.py
-This ensures that thermal and RGB images are correctly paired for model training.

-Step 4: Train the Model
-Start the training process:
-python train.py
-This will use the processed dataset to train the wildfire detection model.


-Dataset
-This project uses the FLAME-2 dataset, which includes multispectral aerial imagery for fire detection. Ensure the dataset is downloaded and extracted correctly before starting.
