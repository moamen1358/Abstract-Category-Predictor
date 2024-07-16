# Abstract Category Predictor

## Overview

The Abstract Category Predictor is a FastAPI application designed to classify sections of academic abstracts into predefined categories using a TensorFlow model. This application utilizes SpaCy for natural language processing and TensorFlow for model loading and prediction. The model is trained to classify abstract sections into one of the following categories: BACKGROUND, CONCLUSIONS, METHODS, OBJECTIVE, and RESULTS.

## Model Performance Metrics

After evaluating the model, the following metrics were obtained:

- **Accuracy:** 82.57%
- **Precision:** 82.43%
- **Recall:** 82.57%
- **F1-score:** 82.47%

## Features

- **FastAPI Integration:** Provides a RESTful API interface for submitting abstracts and receiving category predictions.
- **SpaCy Integration:** Utilizes SpaCy for natural language processing, including sentence segmentation.
- **TensorFlow Model:** Loads a pre-trained model to predict categories based on structural and textual features of abstract sections.
- **Input Preprocessing:** Extracts line numbers, total lines, tokenization, and character-level features to prepare data for model input.
- **Output Format:** Returns a JSON response with each sentence from the input abstract mapped to its predicted category.

## Setup Instructions

1. **Clone Repository:**

   ```bash
   git clone https://github.com/moamen1358/Abstract-Category-Predictor.git
   cd abstract-category-predictor
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Model:**
   Ensure your pre-trained TensorFlow model (`skimlit_tribrid_model/`) is downloaded and placed in the project directory.

   ```bash
   !wget https://storage.googleapis.com/ztm_tf_course/skimlit/skimlit_tribrid_model.zip
   !unzip skimlit_tribrid_model.zip
   ```

4. **Run FastAPI Application:**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Usage:**

   - Submit an abstract to `/predict/` endpoint using a POST request with JSON payload:
     ```bash
     curl -X POST "http://localhost:8000/predict/" -H "Content-Type: application/json" -d '{"abstract": "Your input abstract text here."}'
     ```

6. **Example Response:**
   ```json
   [
     { "line": "First sentence of the abstract.", "label": "BACKGROUND" },
     { "line": "Second sentence of the abstract.", "label": "OBJECTIVE" },
     { "line": "Third sentence of the abstract.", "label": "METHODS" }
   ]
   ```

### Using Postman

1. Open Postman.
2. Create a new request.
3. Set the request method to POST.
4. Enter the URL of your FastAPI application along with the specific endpoint.
   - Example URL: `http://127.0.0.1:8000/predict`
5. Click on the "Body" tab.
6. click on ''raw''.
7. then click on dropdown menu and choose ''json''.
8. Click the "Send" button to send the request to your FastAPI application.

## Example Input

- {"abstract": "This RCT examined the efficacy of a manualized social intervention for children with HFASDs. Participants were randomly assigned to treatment or wait-list conditions. Treatment included instruction and therapeutic activities targeting social skills, face-emotion recognition, interest expansion, and interpretation of non-literal language. A response-cost program was applied to reduce problem behaviors and foster skills acquisition. Significant treatment effects were found for five of seven primary outcome measures (parent ratings and direct child measures). Secondary measures based on staff ratings (treatment group only) corroborated gains reported by parents. High levels of parent, child and staff satisfaction were reported, along with high levels of treatment fidelity. Standardized effect size estimates were primarily in the medium and large ranges and favored the treatment group."}

## Model Construction

### Preprocessing Input

- **Sentence Segmentation:** Uses SpaCy's English model to segment input abstracts into sentences.
- **Feature Extraction:** Extracts line number, total lines, tokenization, and character-level features for each sentence.

### TensorFlow Model

- **Input Preparation:** Converts features into TensorFlow tensors and one-hot encodes categorical features.
- **Model Architecture:** Loads a pre-trained TensorFlow model (`loaded_model`) to predict category labels.

### Prediction

- **Endpoint:** `/predict/` accepts abstract input, preprocesses it, feeds it into the TensorFlow model, and returns predictions in JSON format.

## Contributors

- Mu'min
