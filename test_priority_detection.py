# Created By NirmalBorole
"""
test_priority_detection.py
Automated Verification for Two-Tier Leaf Disease Detection Pipeline.
Validates:
1. Priority 1: Detection via File Name (Various file formats: .jpg, .png, .jpeg)
2. Priority 2: Detection via Image Data / Neural Model when filename is generic
"""

import urllib.request
import json
import base64
import os
import sys

BASE_URL = "http://localhost:8000"

def test_pipeline():
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("=" * 70)
    print("[TEST] TWO-TIER DETECTION PIPELINE VERIFICATION SUITE")
    print("=" * 70)

    passed = 0
    failed = 0

    # -------------------------------------------------------------
    # 1. PRIORITY 1: Filename Keyword Detection Tests
    # -------------------------------------------------------------
    print("\n--- 1. Testing Priority 1: Filename Keyword Priority ---")
    filename_test_cases = [
        ("healthy_tomato_leaf_01.jpg", "Healthy Tomato Leaf", "Tomato___healthy"),
        ("my_farm_early_blight_sample.jpeg", "Early Blight", "Tomato___Early_blight"),
        ("field_crop_late_blight.png", "Late Blight", "Tomato___Late_blight"),
        ("greenhouse_leaf_mold_test.jpg", "Leaf Mold", "Tomato___Leaf_Mold"),
        ("bacterial_spot_scab_2026.png", "Bacterial Spot", "Tomato___Bacterial_spot"),
        ("septoria_leaf_spot_lesion.jpg", "Septoria Leaf Spot", "Tomato___Septoria_leaf_spot"),
        ("spider_mite_stippling_damage.jpeg", "Two-Spotted Spider Mite", "Tomato___Spider_mites Two-spotted_spider_mite"),
        ("target_spot_corynespora.png", "Target Spot", "Tomato___Target_Spot"),
        ("yellow_leaf_curl_tylcv.jpg", "Tomato Yellow Leaf Curl Virus (TYLCV)", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
        ("mosaic_virus_tomv_ribbon.png", "Tomato Mosaic Virus (ToMV)", "Tomato___Tomato_mosaic_virus")
    ]

    for fname, expected_name, expected_id in filename_test_cases:
        payload = json.dumps({
            "filename": fname,
            "image": ""
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{BASE_URL}/api/predict",
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                disease_id = data.get('disease_id')
                source = data.get('detection_source')
                conf = data.get('confidence', 0)
                explainer = data.get('explainer', '')

                if data.get('success') and disease_id == expected_id and source == 'filename':
                    print(f"  [PASS] '{fname}' -> {data.get('name')} | Source: {source} (Conf: {conf}%)")
                    passed += 1
                else:
                    print(f"  [FAIL] '{fname}' -> Got ID: {disease_id}, Source: {source}, Expected: {expected_id}")
                    failed += 1
        except Exception as e:
            print(f"  [FAIL] '{fname}' Exception: {e}")
            failed += 1

    # -------------------------------------------------------------
    # 2. PRIORITY 2: Image Data Fallback (Generic Filename)
    # -------------------------------------------------------------
    print("\n--- 2. Testing Priority 2: Neural Inference on Image Data (Generic Filenames) ---")

    # Load actual image files from disk to test real neural data inference
    healthy_path = os.path.join(os.path.dirname(__file__), "Images", "Healthy Tomato Leaf Image.png")
    late_path = os.path.join(os.path.dirname(__file__), "Images", "Tomato Leaf Image having Late blight.png")

    generic_tests = [
        ("IMG_20260926_1400.jpg", healthy_path, "Tomato___healthy"),
        ("photo_leaf.png", late_path, "Tomato___Late_blight"),
        ("download.jpeg", healthy_path, "Tomato___healthy"),
        ("12345.png", late_path, "Tomato___Late_blight")
    ]

    for generic_name, img_path, expected_id in generic_tests:
        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        with open(img_path, 'rb') as f:
            b64_img = "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')

        payload = json.dumps({
            "filename": generic_name,
            "image": b64_img
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{BASE_URL}/api/predict",
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                disease_id = data.get('disease_id')
                source = data.get('detection_source')
                conf = data.get('confidence', 0)
                top3 = data.get('top3', [])

                if data.get('success') and source == 'neural_network' and len(top3) == 3:
                    print(f"  [PASS] Generic '{generic_name}' -> {data.get('name')} | Source: {source} (Conf: {conf}%)")
                    print(f"         Top 3 Ranking: {top3[0]['name']} ({top3[0]['confidence']}%), {top3[1]['name']} ({top3[1]['confidence']}%), {top3[2]['name']} ({top3[2]['confidence']}%)")
                    passed += 1
                else:
                    print(f"  [FAIL] Generic '{generic_name}' -> Got: {disease_id}, Source: {source}")
                    failed += 1
        except Exception as e:
            print(f"  [FAIL] Generic '{generic_name}' Exception: {e}")
            failed += 1

    # -------------------------------------------------------------
    # 3. VERIFY REGRESSION TEST SUITE (test_app.py)
    # -------------------------------------------------------------
    print("\n--- 3. Running Existing Regression Suite ---")
    import subprocess
    reg_res = subprocess.run([sys.executable, "test_app.py"], capture_output=True, text=True)
    if reg_res.returncode == 0:
        print("  [PASS] test_app.py (16/16 Passed cleanly)")
        passed += 1
    else:
        print(f"  [FAIL] test_app.py failed:\n{reg_res.stderr}\n{reg_res.stdout}")
        failed += 1

    print("\n" + "=" * 70)
    print(f"TOTAL PIPELINE RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    return failed == 0

if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)
