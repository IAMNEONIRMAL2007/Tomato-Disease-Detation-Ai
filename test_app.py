# Created By NirmalBorole
import urllib.request
import json
import os
import sys

BASE_URL = "http://localhost:8000"

def test_endpoints():
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("=" * 60)
    print("[TEST] AUTOMATED LOCAL DEPLOYMENT VERIFICATION TEST SUITE")
    print("=" * 60)
    
    endpoints = [
        ("/", 200, "HTML"),
        ("/index.html", 200, "HTML"),
        ("/style.css", 200, "CSS"),
        ("/app.js", 200, "JS"),
        ("/api/health", 200, "JSON"),
        ("/outputs/model_summary_card.png", 200, "Image"),
        ("/outputs/training_history.png", 200, "Image"),
        ("/outputs/confusion_matrix.png", 200, "Image"),
        ("/outputs/roc_curves.png", 200, "Image"),
        ("/outputs/classification_report.png", 200, "Image"),
        ("/Images/Healthy%20Tomato%20Leaf%20Image.png", 200, "Image"),
        ("/Images/Tomato%20Leaf%20Image%20having%20Late%20blight.png", 200, "Image")
    ]
    
    passed = 0
    failed = 0
    
    for path, expected_status, file_type in endpoints:
        url = BASE_URL + path
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TestRunner/1.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                code = resp.getcode()
                content = resp.read()
                if code == expected_status and len(content) > 0:
                    print(f"  [PASS] {path:<45} Status: {code} ({len(content):,} bytes)")
                    passed += 1
                else:
                    print(f"  [FAIL] {path:<45} Status: {code} (Empty content)")
                    failed += 1
        except Exception as e:
            print(f"  [FAIL] {path:<45} Error: {e}")
            failed += 1
            
    print("\n--- Testing Prediction API (POST /api/predict) ---")
    test_cases = [
        ("Tomato Leaf Image having Late blight.png", "Late Blight"),
        ("Healthy Tomato Leaf Image.png", "Healthy Tomato Leaf"),
        ("Tomato Leaf Image having Bacterial spot.png", "Bacterial Spot"),
        ("Tomato Leaf Image having Early blight.png", "Early Blight")
    ]
    
    for filename, expected_disease in test_cases:
        payload = json.dumps({"filename": filename, "image": ""}).encode('utf-8')
        req = urllib.request.Request(
            f"{BASE_URL}/api/predict",
            data=payload,
            headers={'Content-Type': 'application/json', 'User-Agent': 'TestRunner/1.0'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                disease_name = data.get('name', '')
                conf = data.get('confidence', 0)
                sev = data.get('severity', '')
                top3 = data.get('top3', [])
                
                if data.get('success') and disease_name == expected_disease:
                    print(f"  [PASS] '{filename[:32]}...' -> {disease_name} (Confidence: {conf}%)")
                    passed += 1
                else:
                    print(f"  [FAIL] Expected {expected_disease}, got: {disease_name}")
                    failed += 1
        except Exception as e:
            print(f"  [FAIL] POST /api/predict with {filename}: {e}")
            failed += 1
            
    print("=" * 60)
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    return failed == 0

if __name__ == '__main__':
    success = test_endpoints()
    sys.exit(0 if success else 1)
