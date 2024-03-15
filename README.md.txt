Image Sentiment Classifier:

This computer vision project focuses on sentiment classification of images, aiming to discern whether the depicted individuals are expressing happiness or sadness. Leveraging deep learning techniques, specifically convolutional neural networks (CNNs), the project involves training a model to accurately classify images based on their emotional content. The pipeline includes data preprocessing, model development, training, evaluation, and deployment, enabling real-world application of the sentiment classifier. Through this project, we aim to provide a tool for automated sentiment analysis in image data, with potential applications in various fields such as social media analysis, market research, and mood tracking.


1. Install Dependencies and Setup: Install necessary packages and configure GPU memory growth to avoid out-of-memory errors.

2. Remove Dodgy Images: Remove potentially corrupted or mislabeled images from the dataset.

3. Load Data: Load the image dataset from the specified directory and visualize a sample batch of images.

4. Scale Data: Scale the pixel values of the images to the range [0, 1].

5. Split Data: Split the dataset into training, validation, and test sets.

6. Build Deep Learning Model: Define and compile a convolutional neural network (CNN) model for image classification.

7. Train: Train the model on the training data, visualize training and validation performance using TensorBoard.

8. Plot Performance: Plot the training and validation loss, as well as accuracy, to evaluate model performance.

9. Evaluate: Evaluate the trained model on the test data using precision, recall, and binary accuracy metrics.

10. Test: Test the model on a new image to predict its sentiment (happy or sad).

11. Save the Model: Save the trained model for future use.

References
TensorFlow: https://www.tensorflow.org/
OpenCV: https://opencv.org/