# NER + Computer Vision Pipeline for Animal Verification

## Project Overview

This project implements an ML pipeline that combines two different models for two separate tasks:

- **Named Entity Recognition (NER)** — extracting the animal name from a text query
- **Image Classification** — identifying the animal shown in an image

The main goal is to verify whether the user’s text description matches the image content.

The pipeline takes **two inputs**:

1. **Text**
2. **Image**

It then:

1. extracts the animal mention from the text,
2. predicts the animal class from the image,
3. compares both results,
4. returns a boolean output:
   - `True` if the text matches the image
   - `False` otherwise

---

## Example

### Input
- **Text:** `"There is a cow in the picture."`
- **Image:** a cow image

### Pipeline Output
- **Text animal:** `cow`
- **Image animal:** `cow`
- **Match:** `True`

---

## Project Structure

```bash
NER_CV/
│
├── data/
│   ├── animal_test/
│   └── ner/
│
├── models/
│   ├── animals_efficientNetB4/
│   │   ├── best_model.pth
│   │   └── idx_to_class.json
│   │
│   └── ner_bert/
│       ├── best_ner_model.pth
│       ├── id_to_label.json
│       ├── label_to_id.json
│       └── tokenizer/
│
├── notebooks/
│   ├── animal_ner_bert.ipynb
│   ├── animals_cv_train.ipynb
│   └── pipeline.py
│
├── demo.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Models Used

### 1. NER Model
For text processing, the project uses a **BERT-based token classification model**.

Its task is to extract the animal entity from user text.

Examples:
- `"There is a cow in the picture."` → `cow`
- `"I think this is a butterfly."` → `butterfly`

### 2. Computer Vision Model
For image classification, the project uses **EfficientNet-B4**.

This model takes an image as input and predicts the corresponding animal class.

---

## How the Pipeline Works

The pipeline performs the following steps:

1. Receives a text input
2. Receives an image path
3. Runs the NER model to extract the animal mention from the text
4. Runs the CV model to classify the animal in the image
5. Compares both outputs
6. Returns a dictionary with the final result:

```python
{
    "text_animal": "...",
    "image_animal": "...",
    "match": True / False
}
```

---

## Required Model Files

This project requires pretrained model weights and auxiliary mapping files that are **not included in the repository**.

Before running the project, download the full **`models`** folder from Google Drive and place it in the project root directory.

**Google Drive link:**  
https://drive.google.com/drive/folders/1DKUc8H83jgVBVa2hqznFwG5ciVF4D08l?usp=sharing

The `models` folder must contain:

### CV model files
- `models/animals_efficientNetB4/best_model.pth`
- `models/animals_efficientNetB4/idx_to_class.json`

### NER model files
- `models/ner_bert/best_ner_model.pth`
- `models/ner_bert/id_to_label.json`
- `models/ner_bert/label_to_id.json`
- `models/ner_bert/tokenizer/`

Without these files, the pipeline will not be able to load the models for inference.

---

## Installation

It is recommended to create a virtual environment and install dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Or install the main packages manually:

```bash
pip install torch torchvision transformers pillow matplotlib
```

---

## Demo

The main demonstration of the solution is provided in:

```bash
demo.ipynb
```

The demo notebook includes:
- loading trained models,
- loading JSON label mappings,
- running the pipeline on a single example,
- running the pipeline on multiple test examples,
- comparing text and image predictions,
- returning `True` / `False`.

---

## Example Usage

```python
result = run_pipeline(
    text="There is a cow in the picture.",
    image_path=data_path / "cow.jpeg",
    cv_model=cv_model,
    ner_model=ner_model,
    tokenizer=tokenizer,
    idx_to_class=idx_to_class,
    id_to_label=id_to_label,
    device=device
)
```

Expected output:

```python
{
    "text_animal": "cow",
    "image_animal": "cow",
    "match": True
}
```

---

## Main Files

### `pipeline.py`
Contains the core pipeline logic:
- model loading,
- text inference,
- image inference,
- final comparison through `run_pipeline(...)`.

### `demo.ipynb`
Notebook used to demonstrate the complete solution.

### `requirements.txt`
List of project dependencies.

---

## Edge Cases

The demo includes not only correct matching examples, but also different scenarios such as:
- correct text-image match,
- incorrect text-image match,
- different text formulations,
- cases where the NER model fails to extract an animal name from the text.

This helps demonstrate that the solution works beyond a single fixed example.

---

## Technologies Used

- Python
- PyTorch
- torchvision
- Hugging Face Transformers
- EfficientNet-B4
- BERT
- PIL
- matplotlib
- Jupyter Notebook

---

## Project Purpose

This project was created as a test task to demonstrate:
- integration of NLP and Computer Vision in one solution,
- inference pipeline design,
- transformer-based NER usage,
- image classification workflow,
- combining two independent ML models into one boolean verification system.

---

## Author

Oleksandr Novokhatskyi
