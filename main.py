from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from spacy.lang.en import English
from typing import Dict
# Import TensorFlow model dependencies (if needed) - https://github.com/tensorflow/tensorflow/issues/38250
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization





app = FastAPI()

# ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,


# .............................................
# Load the SpaCy model
nlp = English() # setup English sentence parser
sentencizer = nlp.add_pipe("sentencizer")

# Load the TensorFlow model
model_path = "skimlit_tribrid_model/"

# Load downloaded model from Google Storage
loaded_model = tf.keras.models.load_model(model_path)#,
# Class names
class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']

class AbstractInput(BaseModel):
    abstract: str

def split_to_char(text):
    return " ".join(list(text))

def preprocess_input(abstract: str):
    doc = nlp(abstract)
    abstract_lines = [str(sent) for sent in list(doc.sents)]
    total_lines_in_sample = len(abstract_lines)

    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {
            "text": str(line),
            "line_number": i,
            "total_lines": total_lines_in_sample - 1
        }
        sample_lines.append(sample_dict)

    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)

    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    abstract_chars = [split_to_char(sentence) for sentence in abstract_lines]

    return (test_abstract_line_numbers_one_hot,
            test_abstract_total_lines_one_hot,
            tf.constant(abstract_lines),
            tf.constant(abstract_chars))

@app.post("/predict/")
def predict(input_data: AbstractInput):
    inputs = preprocess_input(input_data.abstract)
    predictions = loaded_model.predict(x=(inputs[0], inputs[1], inputs[2], inputs[3]))
    pred_classes = tf.argmax(predictions, axis=1)
    pred_class_names = [class_names[i] for i in pred_classes]

    result = [{"line": line, "label": label} for line, label in zip(inputs[2].numpy(), pred_class_names)]
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
