import json
from pathlib import Path

import torch
from PIL import Image

import torchvision.transforms as T
from torchvision.models import efficientnet_b4
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification

# Define paths to the models and tokenizer
BASE_DIR = Path(__file__).resolve().parent
comp_path = BASE_DIR.parent / "models"
data_path = BASE_DIR.parent / "data"/"animal_test"

model_cv_path = comp_path / "animals_efficientNetB4" / "best_model.pth"
model_ner_path = comp_path / "ner_bert" / "best_ner_model.pth"
tokenizer_path = comp_path / "ner_bert" / "tokenizer"
tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

# Load the index-to-class and id-to-label mappings
with open(comp_path / "animals_efficientNetB4" / "idx_to_class.json", "r") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}

with open(comp_path / "ner_bert" / "id_to_label.json", "r") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}

# Load the cv model
def load_cv_model(model_path, num_classes, device):
    model = efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

# Load the ner models
def load_ner_model(model_path, num_labels, device):
    config = AutoConfig.from_pretrained("bert-base-cased", num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", config=config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

# Predict the animal in the image
def predict_image_animal(model, image_path, idx_to_class, device):
    image = Image.open(image_path).convert("RGB")
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        pred_idx = torch.argmax(outputs, dim=1).item()

    return idx_to_class[pred_idx]

# Predict animals in the text
def predict_text_animal(model, tokenizer, text, id_to_label, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    input_ids = inputs["input_ids"][0].cpu().tolist()
    pred_ids = predictions[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    predicted_labels = [id_to_label[pred_id] for pred_id in pred_ids]

    animal_tokens = []

    for token, label in zip(tokens, predicted_labels):
        if label == "B-ANIMAL":
            animal_tokens.append(token)

    if not animal_tokens:
        return None

    animal_name = tokenizer.convert_tokens_to_string(animal_tokens).strip()
    return animal_name

# Run the pipeline
def run_pipeline(text, image_path, cv_model, ner_model, tokenizer, idx_to_class, id_to_label, device):
    text_animal = predict_text_animal(ner_model, tokenizer, text, id_to_label, device)
    image_animal = predict_image_animal(cv_model, image_path, idx_to_class, device)

    match = text_animal == image_animal

    return {
        "text_animal": text_animal,
        "image_animal": image_animal,
        "match": match
    }




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_cv_classes = len(idx_to_class)
    num_ner_labels = len(id_to_label)

    cv_model = load_cv_model(model_cv_path, num_cv_classes, device)
    ner_model = load_ner_model(model_ner_path, num_ner_labels, device)

    text = "The monarch cow is one of the most recognizable and beloved insects in North America"
    image_path = data_path / "butterfly2.jpeg"

    result = run_pipeline(
        text=text,
        image_path=image_path,
        cv_model=cv_model,
        ner_model=ner_model,
        tokenizer=tokenizer,
        idx_to_class=idx_to_class,
        id_to_label=id_to_label,
        device=device
    )

    print(result)