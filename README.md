# Fine-Tuning on Galaxy Dataset

This project explores the fine-tuning of pre-trained convolutional neural networks (CNNs) to classify galaxy images from the Galaxy Zoo dataset. It also evaluates the impact of applying Low-Rank Approximation (LoRA) to further optimize the model.

## Features
- **Dataset**: Galaxy Zoo dataset, containing labeled galaxy images.
-  **Dataset Link**:  [Galaxy Zoo Dataset](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data) 

- **Model Architecture**: ResNet18 pre-trained CNN, modified and fine-tuned for galaxy classification.
- **Techniques**:
  - Data augmentation (flipping, rotation, resizing) to increase training diversity.
  - Fine-tuning of the CNN's final layers to adapt to the dataset.
  - Implementation of Low-Rank Approximation (LoRA) for model optimization.

## Key Results
- **ResNet without Fine-Tuning**: 68.63% accuracy on test data.
- **ResNet with Fine-Tuning**: Achieved 87.45% accuracy on test data.
- **LoRA Applied**: Accuracy dropped to ~70%, and further investigation is required to address this issue.

## Tools and Libraries
- **Frameworks**: PyTorch, torchvision.
- **Key Libraries**: NumPy, Matplotlib, tqdm, LoRA libraries (`loralib`, `peft`).

## Training Details
- Optimizer: Adam
- Learning Rate: 0.001
- Scheduler: StepLR with decay at every 10 epochs
- Epochs: 10 for fine-tuning
- Validation and testing conducted to monitor accuracy and loss metrics.

## Challenges and Observations
- The model achieved significant improvement through fine-tuning.
- Accuracy dropped after applying LoRA, and further debugging is required to optimize its integration.

## Future Work
- Explore different LoRA configurations and hyperparameters.
- Investigate other fine-tuning techniques and model architectures to improve performance.
