# Object Detection Service

This project provides a RESTful API for object detection using a TensorFlow Lite model. The API allows users to upload garbage images and receive predictions in the form of list detected objects with bounding boxes and confidence scores.

## Table of Contents

- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Virtual Environment](#virtual-environment)
  - [Install Dependencies](#install-dependencies)
- [Project Structure](#project-structure)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)

## Setup

### Prerequisites

- Python 3.6 or higher

### Virtual Environment

1. **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    ```

2. **Activate the Virtual Environment**:

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On macOS and Linux:

      ```bash
      source venv/bin/activate
      ```

### Install Dependencies

 **Install the Required Packages**:

    bash
    pip install -r requirements.txt
    

## Project Structure

```plaintext
.
├── app.py                      # Flask API script
├── models
│       ├── detect.tflite       # TFLite model file
│       └── labelmap.txt        # Label map file
├── uploads                     # Directory to save uploaded images (created automatically)
├── requirements.txt            # List of required Python packages
└── venv                        # Virtual environment directory
```
## Running the Application
1. **Activate the Virtual Environment:**

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS and Linux:
```bash
source venv/bin/activate
```

2. **Run the Flask Application:**

```bash
flask run --host=[host] --port=[port]
```

## API Endpoints
`POST /predict`
- **Description**: Upload an image and receive predictions for detected objects.
- **Request**: 
    - `Content-Type: multipart/form-data`
    - Body: Form-data with an `image` field containing the image file to be uploaded.
- **Response**: JSON object containing detected objects with their confidence scores and bounding box coordinates.
    - example :
        ```
        {
            "detections": [
                {
                    "box": [
                        49,
                        67,
                        284,
                        282
                    ],
                    "confidence": 0.7690085172653198,
                    "object": "plastic"
                }
            ]
        }
        ```
`GET /labels`
- **Description**: Retrieve the list of labels used by the model.
- **Response**:JSON object containing the list of labels.
    - example:
        ```
        {
            "labels": [
                "alumuniumcan",
                "cardboard",
                "paper",
                "plastic",
                "plasticbottle"
            ]
        }
        ```