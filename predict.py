# Created By NirmalBorole
#!/usr/bin/env bash
#!/usr/bin/env python3
"""
🍅 Tomato Disease Prediction - Comprehensive Inference Engine
==============================================================
Classifies tomato leaf images with deep clinical disease pathology,
severity assessment, treatment recommendations, and confidence visualization.

Usage:
    .\\run_predict.bat "Images/Tomato Leaf Image having Late blight.png"
    .\\run_predict.bat "Images/Healthy Tomato Leaf Image.png"
"""

import sys
import os
import json
import glob
import numpy as np

# Suppress verbose TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from PIL import Image

# Default class labels mapped to directory structure
DEFAULT_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Disease Clinical Knowledge Base
DISEASE_DB = {
    "Tomato___Bacterial_spot": {
        "common_name": "Bacterial Spot",
        "pathogen": "Xanthomonas perforans / euvesicatoria (Bacterium)",
        "severity": "HIGH ⚠️",
        "symptoms": "Small, dark, water-soaked circular lesions with yellow chlorotic halos on foliage; fruit shows scab-like dark spots.",
        "cause": "Warm temperatures (24-30°C), heavy rainfall, overhead sprinkler irrigation, and splashing soil.",
        "organic_treatment": "Apply copper octanoate (copper soap) or Bacillus subtilis sprays; prune infected lower leaves immediately.",
        "chemical_treatment": "Fixed copper bactericides combined with Mancozeb to counter copper-resistant strains.",
        "prevention": "Use certified disease-free seeds; avoid overhead irrigation; practice 2-3 year crop rotation without solanaceous crops."
    },
    "Tomato___Early_blight": {
        "common_name": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "MODERATE ⚠️",
        "symptoms": "Brown to black spots with distinct concentric rings ('target-board' pattern) surrounded by yellow chlorosis starting on older lower leaves.",
        "cause": "High humidity, prolonged leaf wetness, and moderate temperatures (24-29°C). Spores overwinter in plant debris.",
        "organic_treatment": "Bio-fungicides like Trichoderma harzianum, neem oil extract, and compost tea foliar sprays.",
        "chemical_treatment": "Chlorothalonil, Azoxystrobin, or Difenoconazole applied at first sign of symptoms.",
        "prevention": "Stake plants for air circulation; mulch soil heavily to stop soil splash; drip irrigate at soil level."
    },
    "Tomato___Late_blight": {
        "common_name": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "CRITICAL 🚨",
        "symptoms": "Large, irregular water-soaked pale green to dark brown rotting lesions. Under humid conditions, white fluffy mold appears on leaf undersides.",
        "cause": "Cool, wet, cloudy weather (15-22°C) with persistent fog or heavy dew. Can destroy entire fields in days.",
        "organic_treatment": "Remove and incinerate/bag infected plants immediately. Apply preventive copper hydroxide before rain events.",
        "chemical_treatment": "Mandipropamid, Cymoxanil, Metalaxyl-M, or Chlorothalonil before or at very onset.",
        "prevention": "Destroy cull piles; avoid planting near potato fields; plant resistant cultivars (e.g., Mountain Magic, Defiant PHR)."
    },
    "Tomato___Leaf_Mold": {
        "common_name": "Leaf Mold",
        "pathogen": "Passalora fulva / Cladosporium fulvum (Fungus)",
        "severity": "MODERATE ⚠️",
        "symptoms": "Pale green to bright yellow spots on upper leaf surface, with velvety olive-green to brown fungal growth underneath.",
        "cause": "High relative humidity (>85%) and warm temperatures (21-24°C). Highly prevalent in greenhouses and high tunnels.",
        "organic_treatment": "Improve greenhouse airflow with exhaust fans; apply copper fungicides or potassium bicarbonate.",
        "chemical_treatment": "Chlorothalonil, Boscalid, or Cyazofamid labeled for tomato leaf mold.",
        "prevention": "Keep greenhouse humidity below 80%; prune lower suckers to maximize cross-ventilation."
    },
    "Tomato___Septoria_leaf_spot": {
        "common_name": "Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici (Fungus)",
        "severity": "HIGH ⚠️",
        "symptoms": "Numerous small, circular spots (2-3 mm) with grayish-white centers and dark brown margins; tiny black fruiting bodies visible inside spots.",
        "cause": "Warm temperatures (20-25°C) coupled with high humidity or rainfall. Overwinters on infected weeds and solanaceous debris.",
        "organic_treatment": "Remove affected bottom leaves; spray copper sulfate or bio-fungicide Bacillus amyloliquefaciens.",
        "chemical_treatment": "Mancozeb, Chlorothalonil, or Quadris (Azoxystrobin) applied on a 7-10 day schedule.",
        "prevention": "Control nightshade weeds; clean all garden stakes with 10% bleach; mulch ground around stem."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "common_name": "Two-Spotted Spider Mite",
        "pathogen": "Tetranychus urticae (Arachnid Pest)",
        "severity": "HIGH ⚠️",
        "symptoms": "Fine yellow/white stippling (speckling) across leaf surface; leaves turn bronze/yellow and dry out; fine webbing visible under leaves.",
        "cause": "Hot, dry, dusty conditions (>30°C, low humidity). Drought stress dramatically accelerates mite reproduction.",
        "organic_treatment": "Release predatory mites (Phytoseiulus persimilis); apply insecticidal soap, neem oil, or horticultural oils to leaf undersides.",
        "chemical_treatment": "Abamectin, Bifenazate (Acramite), or Spiromesifen; rotate modes of action to prevent rapid resistance.",
        "prevention": "Keep plants adequately watered to eliminate drought stress; spray water wash down to knock off mites."
    },
    "Tomato___Target_Spot": {
        "common_name": "Target Spot",
        "pathogen": "Corynespora cassiicola (Fungus)",
        "severity": "MODERATE ⚠️",
        "symptoms": "Small pinpoint brown spots that expand into circular lesions with light brown centers and dark concentric rings; defoliation from center outward.",
        "cause": "Warm humid conditions (20-28°C). Favored by dense foliage canopies and overhead moisture.",
        "organic_treatment": "Bio-fungicides, copper octanoate sprays, and aggressive pruning of crowded foliage.",
        "chemical_treatment": "Pyraclostrobin (Cabrio), Famoxadone + Cymoxanil (Tanos), or Boscalid (Endura).",
        "prevention": "Wider plant spacing (minimum 24 inches); eliminate weed hosts; avoid overhead water."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "common_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Begomovirus (Transmitted by Sweetpotato Whitefly)",
        "severity": "CRITICAL 🚨",
        "symptoms": "Stunted upright plant growth, leaves drastically reduced in size, curling cupped upward, with pronounced chlorotic (yellow) margins.",
        "cause": "Transmitted by the whitefly vector (Bemisia tabaci). Not seed-borne or sap-transmissible.",
        "organic_treatment": "No cure once infected; immediately remove and bag infected plants. Use yellow sticky cards to trap whiteflies; spray insecticidal soap.",
        "chemical_treatment": "Systemic insecticides targeting whiteflies (e.g., Imidacloprid, Dinotefuran, or Spirotetramat).",
        "prevention": "Use 50-mesh insect netting in greenhouses; plant TYLCV-resistant hybrids (e.g., Tygress, Skyway)."
    },
    "Tomato___Tomato_mosaic_virus": {
        "common_name": "Tomato Mosaic Virus (ToMV)",
        "pathogen": "Tobamovirus (Extremely stable mechanical virus)",
        "severity": "HIGH ⚠️",
        "symptoms": "Mottled light and dark green mosaic patterns on leaves; leaves become distorted, blistered, or reduced to 'fern-like' thin ribbons.",
        "cause": "Mechanical transmission through handling, pruning tools, and tobacco products. Survives in seed coats and soil for years.",
        "organic_treatment": "No cure; remove and destroy symptomatic plants. Disinfect hands and shears with nonfat dry milk (20%) or 10% trisodium phosphate.",
        "chemical_treatment": "No chemical viricide exists. Focus exclusively on vector and sanitation control.",
        "prevention": "Wash hands before handling; do not smoke near tomato plants; use certified virus-free seed."
    },
    "Tomato___healthy": {
        "common_name": "Healthy Tomato Leaf",
        "pathogen": "None (Plant is in optimal health)",
        "severity": "OPTIMAL ✅",
        "symptoms": "Vibrant, uniform green foliage with smooth margins, clear venous structure, and no visible necrotic spots or discoloration.",
        "cause": "Balanced nutrition (N-P-K + Calcium), adequate sunlight (6-8 hrs/day), and proper soil moisture management.",
        "organic_treatment": "Maintain balanced organic fertilization (fish emulsion, compost tea, calcium nitrate to prevent blossom end rot).",
        "chemical_treatment": "None required.",
        "prevention": "Maintain regular scouting schedule (twice weekly); continue drip irrigation and mulch practices."
    }
}


def find_best_model(model_dir='models'):
    """Finds the best model file in models directory."""
    if not os.path.exists(model_dir):
        return None

    # 1. Prefer final model
    final_path = os.path.join(model_dir, 'tomato_disease_final.h5')
    if os.path.exists(final_path):
        return final_path

    # 2. Check for intermediate checkpoints
    checkpoints = glob.glob(os.path.join(model_dir, 'tomato_disease_*.h5'))
    if checkpoints:
        # Sort by modification time or filename
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]

    return None


def render_progress_bar(percent, width=28):
    """Draws a visual ASCII progress bar."""
    filled = int(round(width * percent))
    empty = width - filled
    return "█" * filled + "░" * empty


def predict(image_path, model_path=None, model_dir='models', img_size=(224, 224)):
    """Predict disease from a tomato leaf image and display comprehensive diagnostic report."""

    # 1. Validate image
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image not found at path: {image_path}")
        return

    # 2. Locate model
    if model_path is None or not os.path.exists(model_path):
        model_path = find_best_model(model_dir)

    if model_path is None:
        print(f"\n❌ Error: No trained model found in '{model_dir}'")
        print("  → Run `.\\run_train.bat` first to train and generate the model.")
        return

    # 3. Load class names
    class_names = DEFAULT_CLASSES
    class_names_path = os.path.join(model_dir, 'class_names.json')
    if os.path.exists(class_names_path):
        try:
            with open(class_names_path, 'r') as f:
                class_names = json.load(f).get('class_names', DEFAULT_CLASSES)
        except Exception:
            pass

    # 4. Check metadata for custom image dimensions
    metadata_path = os.path.join(model_dir, 'tomato_disease_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                img_size = tuple(metadata['dataset_info']['img_size'])
        except Exception:
            pass

    predictions = None
    tflite_path = os.path.join(model_dir, 'tomato_disease.tflite')

    # Try ultra-fast edge TFLite model first
    if os.path.exists(tflite_path):
        try:
            print(f"\n[+] Loading Quantized Neural Network from: {os.path.basename(tflite_path)}...")
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        except Exception as e:
            print(f"[-] TFLite load note: {e}, attempting Keras load...")

    # Fallback to Keras H5 if TFLite not used
    if predictions is None:
        print(f"\n[+] Loading neural network from: {os.path.basename(model_path)}...")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

        img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        has_internal_norm = any('preprocess' in l.name or 'rescaling' in l.name for l in model.layers)
        if not has_internal_norm:
            img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)[0]

    pred_class_idx = int(np.argmax(predictions))
    confidence = float(predictions[pred_class_idx])
    pred_class_name = class_names[pred_class_idx]

    # Top 3 candidates
    top_3_indices = np.argsort(predictions)[-3:][::-1]

    # Retrieve disease info
    info = DISEASE_DB.get(pred_class_name, {
        "common_name": pred_class_name.replace("Tomato___", "").replace("_", " "),
        "pathogen": "Unknown",
        "severity": "UNKNOWN",
        "symptoms": "N/A",
        "cause": "N/A",
        "organic_treatment": "Consult local agricultural extension.",
        "chemical_treatment": "Consult local agricultural extension.",
        "prevention": "Ensure optimal sanitation and airflow."
    })

    # 7. Print Comprehensive Diagnostic Report
    border = "═" * 74
    divider = "─" * 74

    print("\n" + border)
    print(" 🍅  TOMATO PLANT PATHOLOGY & DISEASE DIAGNOSTIC REPORT")
    print(border)
    print(f"  📁 File Analyzed     : {os.path.basename(image_path)}")
    print(f"  🔬 Model Checkpoint  : {os.path.basename(model_path)}")
    print(f"  📐 Image Dimensions  : {img.size[0]} x {img.size[1]} pixels (scaled to {img_size[0]}x{img_size[1]})")
    print(divider)
    print(f"  🌿 Primary Diagnosis : {info['common_name'].upper()}")
    print(f"  🦠 Pathogen / Type   : {info['pathogen']}")
    print(f"  ⚠️  Severity Level    : {info['severity']}")
    print(f"  🎯 Model Confidence  : {confidence:.2%}  [{render_progress_bar(confidence)}] ")
    print(divider)
    print("  📊 Top 3 Candidate Classifications:")
    for rank, idx in enumerate(top_3_indices, 1):
        c_name = class_names[idx]
        c_disp = DISEASE_DB.get(c_name, {}).get("common_name", c_name.replace("Tomato___", "").replace("_", " "))
        c_prob = predictions[idx]
        marker = "▶" if idx == pred_class_idx else " "
        print(f"     {marker} #{rank} {c_disp:<30} {c_prob:>6.2%}  [{render_progress_bar(c_prob, 18)}]")

    print(divider)
    print("  🔍 Clinical Symptoms:")
    print(f"     {info['symptoms']}")
    print()
    print("  🌱 Biological Cause / Environment:")
    print(f"     {info['cause']}")
    print()
    print("  🧪 Recommended Action Plan:")
    print(f"     • Organic / Biological : {info['organic_treatment']}")
    print(f"     • Chemical Fungicide   : {info['chemical_treatment']}")
    print(f"     • Preventive Measures  : {info['prevention']}")
    print(border)

    if confidence < 0.70:
        print("  ⚠️  ATTENTION: Confidence is below 70%. Please ensure the image")
        print("     is sharply focused and properly framed on a single tomato leaf.")
        print(border)
    print()


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models')

    if len(sys.argv) < 2:
        print(__doc__)
        print("Available test images in ./Images/:")
        img_dir = os.path.join(script_dir, 'Images')
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"  - {f}")
        sys.exit(1)

    predict(sys.argv[1], model_dir=model_dir)
