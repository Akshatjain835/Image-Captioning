# Image Caption Generator

An intelligent image captioning system that automatically generates descriptive text for images using deep learning. This project combines computer vision and natural language processing to create accurate and contextual captions for any image.

## 📋 Overview

This application uses a pre-trained **Xception** neural network for image feature extraction combined with an **LSTM** (Long Short-Term Memory) network for caption generation. The model is trained on the Flickr8k dataset and deployed as an interactive web application using Streamlit.

### Key Features
- 🖼️ **Image Upload**: User-friendly interface to upload images (JPG, JPEG, PNG)
- 🤖 **AI-Powered Captions**: Automatic caption generation using deep learning
- 📊 **Feature Extraction**: State-of-the-art Xception CNN for robust image features
- 💬 **LSTM Sequence Generation**: Advanced LSTM model for natural language caption creation
- 🎨 **Interactive Visualization**: View images with generated captions in real-time

## 🛠️ Requirements

- Python 3.11+
- TensorFlow 2.13.0
- Keras 2.13.1
- Streamlit
- NumPy
- Pandas
- Pillow (PIL)
- Matplotlib

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akshatjain835/Image-Captioning
   cd Image-Captioning
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv\Scripts\activate  # Windows
   source venv/bin/activate      # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install:
   ```bash
   pip install tensorflow==2.13.0 streamlit numpy pandas pillow matplotlib
   ```

## 🚀 Usage

### Running the Application

```bash
streamlit run main.py
```

This will launch the Streamlit web interface at `http://localhost:8501`

### Steps to Generate Captions

1. Open the web application in your browser
2. Click on "Choose an image..." to upload an image
3. The application will:
   - Extract visual features using the Xception model
   - Generate a caption using the LSTM model
   - Display the image with the generated caption

## 📁 Project Structure

```
Image Captioning/
├── main.py                          # Streamlit web application
├── data/
│   ├── Flickr8k.lemma.token.txt    # Captions dataset
│   └── Flickr8k_Dataset/           # Image dataset
├── models/
│   ├── model.keras                 # Caption generation model
│   ├── feature_extractor.keras    # Image feature extraction model
│   └── tokenizer.pkl               # Text tokenizer
├── notebooks/                       # Jupyter notebooks for training
├── .gitignore
└── README.md
```

## 🧠 Model Architecture

### Feature Extraction (Xception)
- **Input**: Image (224×224 pixels)
- **Architecture**: Pre-trained Xception CNN
- **Output**: 2048-dimensional feature vector

### Caption Generation
- **Input**: Image features + previous word sequence
- **Architecture**:
  - Embedding layer (256 dimensions)
  - Bidirectional processing with concatenation
  - LSTM layer (256 units)
  - Dense layers with dropout for regularization
  - Output: Vocabulary probability distribution (8485 words)
- **Max Caption Length**: 34 words

## 📊 Dataset

The project uses the **Flickr8k dataset**:
- **Images**: 8,000 images
- **Captions**: 5 captions per image
- **Vocabulary**: ~8,485 unique words
- **Format**: Lemmatized and tokenized

## ⚙️ Configuration

Key parameters in `main.py`:

```python
max_length = 34          # Maximum caption length
img_size = 224          # Image size for Xception
```

Update model paths if necessary:
```python
model_path = "models/model.keras"
tokenizer_path = "models/tokenizer.pkl"
feature_extractor_path = "models/feature_extractor.keras"
```

## 🔧 Important Notes

### TensorFlow/Keras Compatibility
This project requires **TensorFlow 2.13.0** with **Keras 2.13.1**. This version is compatible with the saved model formats. Do NOT upgrade to TensorFlow 2.15.0 or higher as they use Keras 3.x which is incompatible with these pre-trained models.

If you encounter compatibility issues:
```bash
pip install tensorflow==2.13.0
```

## 📝 Training (Optional)

To retrain the models with your own dataset:

1. Prepare your image-caption dataset
2. Use the notebooks in `notebooks/` directory
3. Update the model paths in `main.py`

## 🎯 Expected Output

The application will display:
- The uploaded image
- A descriptive caption in blue text at the top
- Example: "a woman in a white dress is standing on a beach"

## 🤝 Contributing

Feel free to fork, modify, and improve this project. Some ideas:
- Fine-tune models with different datasets
- Implement beam search for better captions
- Add attention mechanisms
- Support for video captioning

## 📄 License

This project is open source. Please check the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Flickr8k
- **Pre-trained Model**: Xception (Keras Applications)
- **Framework**: TensorFlow/Keras
- **Web Framework**: Streamlit

## 📧 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Happy Captioning!** 🎨✍️
