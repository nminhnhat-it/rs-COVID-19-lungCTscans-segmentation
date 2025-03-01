# COVID-19 Segmentation on Lung CT Scan Images with Improved UNet3+
This project focuses on training a UNet3+ model with a CNN backbone to perform segmentation of COVID-19 infection regions in lung CT scans. By integrating the robust feature extraction of convolutional neural networks (CNNs) with the advanced multi-scale architecture of UNet3+, this application aims to precisely identify and outline infected lung areas. The repository includes all necessary code, configuration files, and instructions to train, fine-tune, or evaluate the model on CT scan datasets.

### Prerequisites
- Python >= 3.11.7.
- pip (Python package manager).
- Optional: GPU support (e.g., NVIDIA CUDA) for faster training.

### Install required libraries 

```bash
pip install -r requirement.txt
```

### Configuration

Create a config.yaml file in the project root to specify settings. Below is an example configuration:

```yaml
batch_size: 16
gradient_accumulation_steps: 1  # Actual batch size = batch_size * gradient_accumulation_steps
num_epochs: 100
learning_rate: 0.00001
data_root_path: dataset/
ckpt_dir: checkpoint/
log_dir: log/
input_shape: 
  - 512
  - 512
  - 3
encoder: resnet  # Options: false, resnet, seresnext
fine_tune_at: false 
augment:  # false: none augment apply
  random_flip: true         
  random_cutout: true       
  random_contrast: true     
  random_brightness: true   
model_summary: false
continue_training: false
training: false
```

### Configuration Variants

- Training from Scratch:
```yaml
continue_training: false
training: true
```

- Continue Training:
```yaml
continue_training: true
training: true
```

- Testing:
```yaml
continue_training: false
training: false
```

### Dataset
The dataset folder structure is show as below:
```
project_root/
├── config.yaml
└── dataset/
    ├── test
    ├── train
    └── val
```

Ensure the `data_root_path` points to your CT scan dataset.

### Usage

Run the application with your configuration file:

```bash
python run.py --config config.yaml
```

- If `training: true`, the model will train and save checkpoints to `ckpt_dir`.
- If `training: false`, it will load a pre-trained model and perform inference.