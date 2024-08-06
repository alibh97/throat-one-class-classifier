# Throat Image Classifier

This repository contains a Flask-based web application that uses a pre-trained Swin Transformer model to classify images as either "Throat" or "Not Throat." The application is containerized using Docker, making it easy to deploy and run on any platform that supports Docker. Additionally, an HTML frontend is provided for users to easily upload images to the app for classification.

This app has been trained on 1181 throat images. One of my teammates gathered this dataset over several months from individuals of various ages. It is still not publicly available. I will site that as soon as it will be available publicly.

---

## Project Structure

    .
    ├── app.py                            # The Flask web application.
    ├── app2.py                           # The second Flask web application which automatically downloads the swinn base model.
    ├── Dockerfile                        # Dockerfile to containerize the application
    ├── requirements.txt                  # Python dependencies
    ├── index.html                        # HTML file for user interface
    ├── local_swin_base_patch4_window7_224/  # Directory containing the pre-trained Swin Transformer model
    │   └── *                             # (model files go here)
    └── models/                           # Directory containing the scaler and One-Class SVM model
        ├── scaler_swinbase.pkl
        └── ocsvm_model_swinbase.pkl
        
## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Git** (optional): If you want to clone the repository directly from GitHub

## How to Set Up the Project

### 1. Clone the Repository

You can clone the repository using Git:

    
    git clone https://github.com/yourusername/throat-image-classifier.git
    cd throat-image-classifier

Alternatively, you can download the repository as a ZIP file and extract it.

### 2. Prepare the Model Files

Ensure that the `local_swin_base_patch4_window7_224/` directory contains the Swin Transformer model files. The directory already contains the `config.json` file and you should manually download and put the `model.safetensors` file in this directory from [this link](https://drive.google.com/uc?id=1cNAfozCGrLhIBmM1XrZHycJb0olG-21F&export=download)
Or you can use `app2.py` instead of `app.py`(default). For that, you should change the last line of `dockerfile` and replace `app.py` with `app2.py`. This makes the app to be able to automatically download and use the model and so there is no need to manually download the model and place it in the `local_swin_base_patch4_window7_224/` directory.

Ensure that the `models/` directory contains the following:

- `scaler_swinbase.pkl`: The scaler used to normalize the image features
- `ocsvm_model_swinbase.pkl`: The One-Class SVM model used for classification

### 3. Build the Docker Image

Navigate to the directory containing the `Dockerfile` and run the following command to build the Docker image:

    docker build -t throat-classifier .

This command will create a Docker image named `throat-classifier` using the instructions provided in the `Dockerfile`.

### 4. Run the Docker Container

Once the image is built, you can run the container using:

    docker run -p 5000:5000 throat-classifier
    

This command will start the Flask web application inside a Docker container and expose it on port `5000`. The application will be accessible at `http://localhost:5000`.

## Using the HTML Frontend

An `index.html` file is provided as a simple user interface to interact with the Flask API.

### 1. Accessing the HTML Page

Once the Docker container is running, open the `index.html` file in your web browser. You can do this by simply double-clicking the file or dragging it into an open browser window.

### 2. Uploading an Image

- **Take a Picture**: Click the "Take Pic" button to use your device's camera to capture an image.
- **Browse for a Picture**: Click the "Browse Pic" button to upload an image from your device.

### 3. View Results

Once the image is uploaded, the app will automatically send it to the Flask API, and the prediction result ("Throat" or "Not Throat") will be displayed on the webpage.

## API Usage

You can also test the API directly by sending a POST request with an image file. For example, using `curl`:

    curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:5000/predict
    

The API will respond with a JSON object indicating whether the image is classified as "Throat" or "Not Throat."

### Sample Response

    {
      "result": "Throat"
    }

or

    {
      "result": "NotThroat"
    }

## Additional Notes

- **CORS Support**: The application includes CORS support, allowing cross-origin requests from any domain.
- **Device**: The model is configured to run on CPU (`device = 'cpu'`). If you wish to use a GPU, you can modify the `device` configuration in `app.py`.

## Troubleshooting

- **Docker Pull Rate Limit**: If you encounter an error related to Docker's pull rate limit, consider logging in to Docker Hub using `docker login` or wait for the rate limit to reset.
- **Port Conflicts**: If port `5000` is already in use on your machine, you can map the container to a different port by changing the `-p` option in the `docker run` command, e.g., `-p 8080:5000`.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

