# 🧠 LSTM Next Word Predictor

An interactive **Next-Word Prediction and Text Generation** application built using **Deep Learning, Natural Language Processing (NLP), TensorFlow, Keras, and Streamlit**.

The application takes a sequence of words as input and uses a trained **Long Short-Term Memory (LSTM)** neural network to predict the next word. It can also generate multiple words based on a starting sentence.

---

## 🚀 Project Overview

Natural language models can learn patterns and relationships between words from text data.

In this project, I developed an LSTM-based language model using a dataset containing **3,038 quotes** and their authors.

The workflow includes:

**Text Data → Preprocessing → Tokenization → Sequence Creation → Padding → LSTM Model → Next-Word Prediction → Text Generation**

The trained model is then integrated into a **Streamlit web application** to provide an interactive user interface.

---

## ✨ Features

* 🔤 Next-word prediction
* ✍️ Multi-word text generation
* 🧠 LSTM-based deep learning model
* 📚 Text tokenization using Keras Tokenizer
* 📏 Sequence padding for model input
* 🎨 Interactive Streamlit interface
* ⚡ Real-time predictions
* 📊 Model information displayed in the application
* 💻 Simple and user-friendly web interface

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* NumPy

### Natural Language Processing

* Text preprocessing
* Tokenization
* Sequence generation
* Padding
* Next-word prediction

### Deployment / UI

* Streamlit

---

## 🧠 Model Architecture

The project uses a neural network based on an **LSTM architecture**.

The general workflow is:

```text
Input Text
    ↓
Text Tokenization
    ↓
Numerical Sequence
    ↓
Sequence Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dense Layer
    ↓
Softmax Prediction
    ↓
Next Word
```

LSTM networks are useful for sequence-based problems because they can learn patterns from previous words and use that context when making predictions.

---

## 📂 Dataset

The model was trained using a quote dataset containing:

* **3,038 quotes**
* Quote text
* Author information

The text was converted to lowercase and punctuation was removed during preprocessing.

The tokenizer vocabulary contains approximately **8,978 words**.

---

## 🔄 Project Workflow

### 1. Data Loading

The quote dataset is loaded using Pandas.

```python
df = pd.read_csv("qoute_dataset.csv")
```

### 2. Text Preprocessing

The quote text is converted to lowercase and punctuation is removed.

### 3. Tokenization

Keras `Tokenizer` is used to convert words into numerical representations.

### 4. Sequence Creation

Text sequences are created so that the model can learn relationships between previous words and the next word.

### 5. Padding

Sequences are padded to a consistent length before being provided to the neural network.

### 6. Model Training

The LSTM model is trained for **10 epochs** with a batch size of **32**.

The training accuracy increased from approximately **5% in the first epoch to approximately 27% by the tenth epoch**.

### 7. Prediction

For a given input sequence, the model predicts the most probable next word.

### 8. Text Generation

The predicted word can be added back to the input sequence and the process can be repeated to generate additional text.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit interface called **LSTM Word Studio**.

The application provides two main functions:

### 🔮 Next Word Prediction

Enter a sentence such as:

```text
what are you
```

The model predicts a possible next word.

### ✨ Text Generation

Provide a starting sentence such as:

```text
are you a
```

and select the number of words to generate.

The application then repeatedly predicts the next word to create a longer sequence.

---

## 📁 Project Structure

```text
lstm-next-word-predictor/
│
├── app.py
├── lstm_model.keras
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File               | Description                   |
| ------------------ | ----------------------------- |
| `app.py`           | Streamlit application         |
| `lstm_model.keras` | Trained LSTM model            |
| `tokenizer.pkl`    | Saved Keras tokenizer         |
| `max_len.pkl`      | Saved maximum sequence length |
| `requirements.txt` | Required Python packages      |
| `README.md`        | Project documentation         |
| `.gitignore`       | Files excluded from Git       |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/lstm-next-word-predictor.git
```

Navigate to the project directory:

```bash
cd lstm-next-word-predictor
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL where the application can be opened in your browser.

---

## 📦 Required Files

The application requires the following trained model files:

```text
lstm_model.keras
tokenizer.pkl
max_len.pkl
```

These files should be located in the same directory as `app.py`.

---

## 📈 Training

The model was trained using:

```text
Epochs: 10
Batch Size: 32
```

Training accuracy improved progressively during training:

| Epoch | Accuracy |
| ----: | -------: |
|     1 |    4.99% |
|     2 |    9.04% |
|     3 |   11.10% |
|     4 |   13.16% |
|     5 |   14.88% |
|     6 |   16.42% |
|     7 |   18.33% |
|     8 |   20.81% |
|     9 |   23.58% |
|    10 |   26.87% |

This project was primarily created to understand the complete workflow of building a sequence-based deep learning application, from text preprocessing and model training to interactive deployment.

---

## 🎯 Learning Objectives

Through this project, I practiced:

* Python for machine learning
* Text preprocessing
* NLP fundamentals
* Tokenization
* Sequence modelling
* LSTM neural networks
* TensorFlow and Keras
* Model saving and loading
* Streamlit application development
* Building an end-to-end ML application

---

## 🔮 Future Improvements

Potential improvements include:

* Improve model accuracy with a larger dataset
* Experiment with different LSTM architectures
* Add Bidirectional LSTM
* Compare LSTM with GRU
* Add top-k prediction probabilities
* Improve text-generation quality
* Experiment with Transformer-based models
* Deploy the application online
* Add more evaluation metrics

---

## 👨‍💻 Author

**Parth Vaghasiya**

Master's Student – Smart Energy Systems
Germany

Interested in:

* Artificial Intelligence
* Machine Learning
* Generative AI
* NLP
* Deep Learning
* Data Analytics
* Energy & AI

---

## ⭐ If you found this project useful

Feel free to explore the repository, experiment with the model, and provide feedback.
