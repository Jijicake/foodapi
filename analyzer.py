# analyzer.py

import csv, ollama, re, time, torch, os
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

variation_map = {}
ingredient_data = {}
gemma_cache = {}
health_scores = {
    "Safe": 100,
    "Cut Back": 75,
    "Certain People Should Avoid": 50,
    "Caution": 20,
    "Avoid": -25,
    "Stop": -100
}

def load_ingredient_health(file_path):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return {row[0].lower(): row for row in reader if row}

def load_variations(file_path):
    v_map = {}
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for primary, variation in reader:
            v_map[variation.lower()] = primary.lower()
            v_map[primary.lower()] = primary.lower()
    return v_map

def load_gemma_cache(path="gemma_cache.csv"):
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            return {row[0]: row[1] for row in reader if len(row) == 2}
    return {}

def save_gemma_cache(cache, path="gemma_cache.csv"):
    with open(path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for key, value in cache.items():
            writer.writerow([key.lower(), value])

def ask_gemma(user_input, known_ingredients):
    ingredient_list = ', '.join(ing[0] for ing in known_ingredients.values())
    prompt = (
        f"A user asked about '{user_input}', but it's not in the list of known ingredients.\n"
        f"Here is the list: [{ingredient_list}]\n\n"
        f"Return the best match from the list, based on meaning. No explanation."
    )
    response = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response['message']['content'].strip()
    gemma_cache[user_input] = result
    return result

def extract_ingredients(text_block):
    prompt = (
        f"The following is a list of text from a food label:\n\n'{text_block}'\n\n"
        "Please extract and list the **ingredients only**, separated by commas. "
        "Do not include anything else."
    )
    response = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}]
    )
    return [i.strip() for i in response['message']['content'].strip().split(",")]

# Main analyzer function
def analyze_image(img: Image.Image, cut_back=None, penalty_override=None):
    global variation_map, ingredient_data, gemma_cache

    variation_map = load_variations("ingredient_variations.csv")
    ingredient_data = load_ingredient_health("chemical_cuisine_additives.csv")
    gemma_cache = load_gemma_cache()

    results = recognition_predictor([img], [["en"]], detection_predictor)
    text_lines = results[0].text_lines
    only_text = [line.text for line in text_lines]
    text_block = "\n".join(only_text)
    ingredients = extract_ingredients(text_block)

    healthy = []
    actual_ingredients = []
    penalty = -100 if not penalty_override else -int(penalty_override)

    def slugify(text, h):
        healthy.append(h)
        actual_ingredients.append(text)

    def basic_match(user_input):
        return ingredient_data.get(user_input.lower())

    def search(prompt):
        base = gemma_cache.get(prompt.lower(), prompt.lower())
        base = variation_map.get(base, base)
        match = None
        for key in ingredient_data:
            if base in key:
                match = ingredient_data[key]
                break
        if match:
            slugify(match[0], match[1])
        else:
            suggestion = ask_gemma(base, ingredient_data)
            match = basic_match(suggestion)
            if match:
                slugify(match[0], match[1])
            else:
                healthy.append("Safe")
                actual_ingredients.append(suggestion)

    for ing in ingredients:
        search(ing)

    g = 0
    if cut_back:
        for cb in cut_back:
            cb_clean = cb.strip().lower()
            base_cb = variation_map.get(cb_clean, cb_clean)
            match = basic_match(base_cb)
            matched_cut = match[0] if match else ask_gemma(base_cb, ingredient_data)
            for ing in actual_ingredients:
                if matched_cut.lower() in ing.lower():
                    healthy.append("Stop")
                    g += 1

    health_scores["Stop"] = penalty
    score = sum(health_scores.get(h, 0) for h in healthy)
    final_score = score / max(len(healthy), 1)

    save_gemma_cache(gemma_cache)
    return {
        "ingredients": actual_ingredients,
        "tags": healthy,
        "score": final_score,
        "raw_ocr": only_text
    }
