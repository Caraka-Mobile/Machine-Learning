# Bangkit-Machine-Learning

This repository contains resources used during the capstone project for Bangkit Machine Learning. The project focuses on building machine learning models for our application. Specifically, we have built image classifier models for Javanese scripts.

<div style="display: flex; justify-content: space-between;">
    <img src="./overview/Layer 2.png" width="400" height="700">
    <img src="./overview/Layer 7.png" width="400" height="700">
</div>

## Architecture

![CNN Architecture](./overview/cnn.png)

_Sources: [upgrad.com](https://www.upgrad.com/blog/basic-cnn-architecture/)_

## Datasets

- [Javanese_Script_Classification](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- [Aksara_Jawa](https://www.kaggle.com/datasets/phiard/aksara-jawa/data)
- [Rice Leaf Disease Dataset](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)
- [Soil Type Dataset](https://www.kaggle.com/datasets/prasanshasatpathy/soil-types)

## Models

### Model Overview

The model architecture for classifying Javanese script consists of the following layers:

- **Convolutional Layer:**

  - 32 filters with a 3x3 kernel size.
  - ReLU activation function.
  - Input shape of (160, 160, 3) for the first layer to match the image size and color channels.

- **Batch Normalization Layer:**

  - Normalizes the outputs of the previous layer to improve training stability.

- **Max Pooling Layer:**

  - Pool size of 2x2 to reduce the dimensionality and computational load.

- **Dropout Layer:**

  - Dropout rate of 25% to prevent overfitting.

- **Convolutional Layer:**

  - 64 filters with a 3x3 kernel size.
  - ReLU activation function.

- **Batch Normalization Layer:**

  - Normalizes the outputs of the previous layer.

- **Max Pooling Layer:**

  - Pool size of 2x2.

- **Dropout Layer:**

  - Dropout rate of 30%.

- **Convolutional Layer:**

  - 128 filters with a 3x3 kernel size.
  - ReLU activation function.

- **Batch Normalization Layer:**

  - Normalizes the outputs of the previous layer.

- **Max Pooling Layer:**

  - Pool size of 2x2.

- **Dropout Layer:**

  - Dropout rate of 40%.

- **Convolutional Layer:**

  - 256 filters with a 3x3 kernel size.
  - ReLU activation function.

- **Batch Normalization Layer:**

  - Normalizes the outputs of the previous layer.

- **Max Pooling Layer:**

  - Pool size of 2x2.

- **Dropout Layer:**

  - Dropout rate of 50%.

- **Convolutional Layer:**

  - 512 filters with a 3x3 kernel size.
  - ReLU activation function.

- **Batch Normalization Layer:**

  - Normalizes the outputs of the previous layer.

- **Max Pooling Layer:**

  - Pool size of 2x2.

- **Dropout Layer:**

  - Dropout rate of 50%.

- **Flatten Layer:**

  - Converts the 2D feature maps into a 1D feature vector.

- **Dense Layer:**

  - 512 units with ReLU activation function.
  - L2 regularization with a factor of 0.001 to prevent overfitting.

- **Batch Normalization Layer:**

  - Normalizes the outputs of the previous layer.

- **Dropout Layer:**

  - Dropout rate of 50%.

- **Output Dense Layer:**
  - 20 units, one for each class of Javanese script.
  - Softmax activation function to output probability distribution over the classes.

This convolutional neural network (CNN) model is designed to effectively classify Javanese script by leveraging multiple layers of convolutional operations, normalization, and dropout to improve generalization and prevent overfitting. The model culminates in a dense layer with softmax activation, providing a probability distribution over the 20 classes for classification.

### Data Processing

The dataset used for the `Javanese_Script_Classification` model consists of images belonging to twenty different classes of Javanese script characters. These classes represent different characters from the traditional Javanese script, each requiring accurate recognition and classification by the model.

Data augmentation techniques are applied to increase the diversity and size of the dataset. The `image_dataset_from_directory` function from TensorFlow is used for loading and splitting the data into training and validation sets with a validation split of 20%. Various transformations such as rescaling, rotation, zooming, flipping, shifting, shearing, and brightness adjustments are applied to the images to enhance the model's robustness and generalization capability.

### Model Training

Training is performed for 500 epochs with a batch size of 50. Two callbacks, `EarlyStopping` and `ReduceLROnPlateau`, are used to enhance the training process. The `EarlyStopping` callback monitors the validation loss and stops training if the loss does not improve for 30 consecutive epochs, restoring the best weights. The `ReduceLROnPlateau` callback reduces the learning rate by a factor of 0.2 if the validation loss does not improve for 10 consecutive epochs, with a minimum learning rate of 0.0001.

### Model Evaluation

The trained `Javanese_Script_Classification` model is evaluated using a separate test dataset. The evaluation provides the loss and accuracy scores of the model on the test dataset, offering insights into its performance on unseen data.

Additionally, a sample of images from the test dataset is used to demonstrate the model's prediction capabilities. This involves visualizing the input images along with the predicted and actual class labels to illustrate the model's effectiveness and any potential areas for improvement.

### Model Saving and Conversion

The trained model is saved in the HDF5 format as `model.h5` for future use. To integrate the model with Android applications, the model is converted to the TensorFlow Lite (TFLite) format using the TFLite Converter. The TFLite model is saved as `model.tflite` for deployment on resource-constrained devices.

## Requirements

To run the notebook and utilize the model, the following dependencies are required:

- TensorFlow
- TensorFlow Lite
- Keras
- Matplotlib
- NumPy
- PIL

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Caraka-Mobile/Machine-Learning.git
   ```

2. Install the required dependencies in your Google Colab/Jupyter Notebook:

   ```bash
   pip install tensorflow keras matplotlib numpy pillow psutil
   ```

3. Navigate to the repository `Notebooks` directory and open the notebooks.

4. Run the cells in the notebook to train the model and evaluate its performance.
5. Save the trained model as `model.h5` for future use.
6. Convert the model to TFLite format using the provided code and save it as `model.tflite`.

## Contributing

Contributions to this project are welcome. If you have any ideas, suggestions, or improvements, please submit a pull request. Make sure to follow the existing coding style and guidelines.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute the code as per the license terms.
