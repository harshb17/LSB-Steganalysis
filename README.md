# LSB Steganography Detection (Multi-Payload)

This project is an AI-powered web demo for detecting steganographic payloads in PNG images using advanced Least Significant Bit (LSB) feature extraction and PyTorch deep learning models. It supports both binary classification (clean vs stego) and multi-class payload detection for various hidden data types.

## 🚀 Features

- **Upload PNG image** and instantly check for hidden payloads
- **Binary detection:** Clean (`🔓`) vs Stego (`🔒`)
- **Multi-class payload detection:** HTML, JS, PS, Clean
- **Fast AI-powered feature extraction (LSB, entropy, randomness, URL pattern)**
- **Interactive web UI built with Flask + Bootstrap**
- **Visual probabilities and confidence bars**
- **Secure server-side PyTorch model inference**

## 🧠 Model Architecture

- Binary & multi-class classifiers: Deep Neural Networks using PyTorch
- LSB feature extraction (2 LSBs of all RGB pixels, plus statistical features)
- Scaler and label encoders via scikit-learn
- Trained on thousands of genuine and stego payload images

## 📝 How It Works

- **Upload an image**
- Client sends the image to backend `/predict` endpoint
- Backend extracts features, normalizes & feeds them into binary and multi-class models
- Results returned as JSON and shown in an interactive UI

## ⚙️ Setup

```bash
pip install -r requirements.txt
python app.py
```

The demo runs at `http://localhost:5000`.

## 📦 Model Artifacts

The backend expects `final_bin_best.pth`, `final_multi_best.pth`, `final_scaler.pkl`, `final_bin_le.pkl`, and `final_multi_le.pkl` in the project root. The artifact files committed here are Git LFS pointer stubs; place the actual trained artifacts in the project root before running the app.

## 📚 About

Steganography is the science of hiding secret data within digital files. This demo exposes hidden payloads embedded via LSB manipulation using AI.  
Made by [Harsh Bhoi](https://github.com/harshb17).

---

## License

Free for academic and personal demo use.

## Contact

For collaboration or research inquiries, open an Issue or contact via GitHub.
