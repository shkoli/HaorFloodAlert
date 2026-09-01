"""
extract_model_info.py
=====================
তোমার HaorFloodAlert project থেকে
review paper এর জন্য দরকারি সব info বের করে।

Run: python extract_model_info.py
Output: model_info.txt
"""

import os
import json
import glob

INFO = {}

# ─── 1. Python files থেকে model info খোঁজো ───
def scan_python_files():
    findings = []
    py_files = glob.glob("**/*.py", recursive=True)
    
    keywords = [
        "accuracy", "f1", "precision", "recall",
        "n_estimators", "learning_rate", "max_depth",
        "RandomForest", "XGBClassifier", "ensemble",
        "NDWI", "TWI", "SAR", "alpha",
        "train_test_split", "cross_val"
    ]
    
    for filepath in py_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                for kw in keywords:
                    if kw.lower() in line.lower():
                        findings.append({
                            "file": filepath,
                            "line": i+1,
                            "content": line.strip()
                        })
                        break
        except:
            pass
    
    return findings


# ─── 2. CSV/JSON result files খোঁজো ───
def scan_result_files():
    results = {}
    
    # CSV files
    csv_files = glob.glob("**/*.csv", recursive=True)
    for f in csv_files:
        results[f] = "CSV file found"
    
    # JSON files
    json_files = glob.glob("**/*.json", recursive=True)
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            results[f] = str(data)[:200]
        except:
            results[f] = "JSON file found (could not parse)"
    
    # .pkl model files
    pkl_files = glob.glob("**/*.pkl", recursive=True)
    for f in pkl_files:
        results[f] = "Saved model file (.pkl)"
    
    return results


# ─── 3. Markdown/README থেকে info খোঁজো ───
def scan_markdown_files():
    findings = {}
    md_files = glob.glob("**/*.md", recursive=True)
    
    for filepath in md_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Key sections খোঁজো
            keywords = [
                "accuracy", "f1", "precision", "recall",
                "feature", "model", "result", "performance",
                "NDWI", "TWI", "SAR", "lead time",
                "dataset", "training", "validation", "test"
            ]
            
            relevant_lines = []
            for line in content.split("\n"):
                for kw in keywords:
                    if kw.lower() in line.lower():
                        relevant_lines.append(line.strip())
                        break
            
            if relevant_lines:
                findings[filepath] = relevant_lines[:20]
        except:
            pass
    
    return findings


# ─── 4. Saved model থেকে info extract ───
def extract_from_saved_model():
    info = {}
    
    pkl_files = glob.glob("**/*.pkl", recursive=True)
    if not pkl_files:
        return {"status": "No saved model (.pkl) found"}
    
    try:
        import pickle
        import numpy as np
        
        for pkl_path in pkl_files[:3]:  # Max 3 models check
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            
            model_info = {"file": pkl_path}
            
            # Model type
            model_info["type"] = type(model).__name__
            
            # Hyperparameters
            if hasattr(model, "get_params"):
                model_info["params"] = model.get_params()
            
            # Feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                model_info["feature_importances"] = importances.tolist()
                model_info["n_features"] = len(importances)
            
            # Classes
            if hasattr(model, "classes_"):
                model_info["classes"] = model.classes_.tolist()
            
            info[pkl_path] = model_info
    
    except ImportError:
        info["error"] = "pickle/numpy not available"
    except Exception as e:
        info["error"] = str(e)
    
    return info


# ─── 5. Notebook থেকে results খোঁজো ───
def scan_notebooks():
    findings = {}
    nb_files = glob.glob("**/*.ipynb", recursive=True)
    
    for filepath in nb_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                nb = json.load(f)
            
            outputs = []
            for cell in nb.get("cells", []):
                # Output cells
                for output in cell.get("outputs", []):
                    text = output.get("text", [])
                    if isinstance(text, list):
                        text = "".join(text)
                    
                    # Accuracy/metrics খোঁজো
                    if any(kw in text.lower() for kw in 
                           ["accuracy", "f1", "precision", "recall", "auc"]):
                        outputs.append(text[:300])
            
            if outputs:
                findings[filepath] = outputs
        except:
            pass
    
    return findings


# ─── MAIN ───
def main():
    print("\n🔍 HaorFloodAlert — Model Info Extractor")
    print("=" * 50)
    print(f"Scanning: {os.getcwd()}")
    print()
    
    report = []
    report.append("# HaorFloodAlert — Extracted Model Info")
    report.append("=" * 50)
    report.append(f"Scanned from: {os.getcwd()}\n")
    
    # 1. Python files
    print("→ Scanning Python files...")
    py_findings = scan_python_files()
    report.append("\n## 1. Key Lines from Python Files")
    if py_findings:
        for item in py_findings[:30]:  # Max 30 lines
            report.append(f"  [{item['file']} : line {item['line']}]")
            report.append(f"  {item['content']}")
    else:
        report.append("  No Python files found")
    
    # 2. Result files
    print("→ Scanning result files (CSV/JSON/PKL)...")
    result_files = scan_result_files()
    report.append("\n## 2. Result Files Found")
    if result_files:
        for f, info in result_files.items():
            report.append(f"  ✓ {f}")
            report.append(f"    {info[:100]}")
    else:
        report.append("  No result files found")
    
    # 3. Markdown files
    print("→ Scanning README/markdown files...")
    md_findings = scan_markdown_files()
    report.append("\n## 3. Key Info from Markdown Files")
    if md_findings:
        for filepath, lines in md_findings.items():
            report.append(f"\n  [{filepath}]")
            for line in lines:
                if line:
                    report.append(f"  {line}")
    else:
        report.append("  No markdown files found")
    
    # 4. Saved models
    print("→ Checking saved model files...")
    model_info = extract_from_saved_model()
    report.append("\n## 4. Saved Model Info")
    for key, val in model_info.items():
        report.append(f"  {key}: {str(val)[:200]}")
    
    # 5. Notebooks
    print("→ Scanning Jupyter notebooks...")
    nb_findings = scan_notebooks()
    report.append("\n## 5. Notebook Outputs (Metrics)")
    if nb_findings:
        for filepath, outputs in nb_findings.items():
            report.append(f"\n  [{filepath}]")
            for out in outputs:
                report.append(f"  {out[:200]}")
    else:
        report.append("  No notebooks found")
    
    # Save report
    report_text = "\n".join(report)
    output_file = "model_info.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\n✅ Done! Saved → {output_file}")
    print("\n📋 Quick Summary:")
    print(f"   Python files scanned: {len(glob.glob('**/*.py', recursive=True))}")
    print(f"   Result files found: {len(result_files)}")
    print(f"   Markdown files: {len(md_findings)}")
    print(f"   Notebooks: {len(nb_findings)}")
    print(f"\n👉 Open model_info.txt to see all extracted info")
    print("   Then copy relevant parts into your review paper prompt\n")


if __name__ == "__main__":
    main()
