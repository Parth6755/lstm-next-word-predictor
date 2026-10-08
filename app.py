import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LSTM Word Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(99,102,241,0.16), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(168,85,247,0.13), transparent 30%),
        #080b14;
    color: #f8fafc;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main container */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hero */
.hero {
    padding: 45px 40px;
    border-radius: 28px;
    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.92),
            rgba(15,23,42,0.88)
        );
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow: 0 30px 80px rgba(0,0,0,0.35);
    margin-bottom: 30px;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(129,140,248,0.28);
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 48px;
    line-height: 1.05;
    margin: 0;
    font-weight: 800;
    letter-spacing: -2px;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #818cf8,
        #c084fc,
        #f472b6
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #94a3b8;
    font-size: 17px;
    max-width: 720px;
    line-height: 1.7;
    margin-top: 18px;
}

/* Cards */
.card {
    padding: 25px;
    border-radius: 22px;
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.12);
    box-shadow: 0 20px 50px rgba(0,0,0,0.20);
    margin-bottom: 20px;
}

.card-title {
    font-size: 14px;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

.card-value {
    font-size: 28px;
    font-weight: 800;
}

/* Prediction result */
.prediction {
    margin-top: 18px;
    padding: 25px;
    border-radius: 20px;
    background:
        linear-gradient(
            135deg,
            rgba(79,70,229,0.18),
            rgba(168,85,247,0.12)
        );
    border: 1px solid rgba(129,140,248,0.25);
}

.predicted-word {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #818cf8,
        #c084fc
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Output */
.output-box {
    padding: 30px;
    border-radius: 22px;
    background: rgba(2,6,23,0.8);
    border: 1px solid rgba(148,163,184,0.12);
    font-size: 22px;
    line-height: 1.8;
    color: #e2e8f0;
    min-height: 120px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 12px 20px;
    font-weight: 700;
    border: 1px solid rgba(129,140,248,0.3);
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );
    color: white;
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(99,102,241,0.30);
}

/* Text area */
textarea {
    background: #0f172a !important;
    color: #f8fafc !important;
    border-radius: 16px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080b14;
    border-right: 1px solid rgba(148,163,184,0.08);
}

section[data-testid="stSidebar"] h2 {
    font-weight: 800;
}

.metric-card {
    text-align: center;
    padding: 20px;
    border-radius: 18px;
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(148,163,184,0.10);
}

.metric-number {
    font-size: 26px;
    font-weight: 800;
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 40px 0 10px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "lstm_model.keras"
    )

    return model


@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer


@st.cache_resource
def load_max_len():

    with open("max_len.pkl", "rb") as file:
        max_len = pickle.load(file)

    return max_len


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model = load_model()
    tokenizer = load_tokenizer()
    max_len = load_max_len()

    model_loaded = True

except Exception as e:

    model_loaded = False
    model = None
    tokenizer = None
    max_len = None


# ============================================================
# FUNCTIONS
# ============================================================

def clean_text(text):

    text = text.lower()

    # Match the preprocessing used in the notebook
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~]', '', text)

    return text


def predict_next_word(text):

    text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return None, 0

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_word = None

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            predicted_word = word
            break

    confidence = float(
        prediction[predicted_index]
    )

    return predicted_word, confidence


def generate_text(seed_text, n_words):

    generated = clean_text(seed_text)

    for _ in range(n_words):

        word, confidence = predict_next_word(
            generated
        )

        if not word:
            break

        generated += " " + word

    return generated


def get_top_predictions(text, top_k=5):

    text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return []

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )[0]

    top_indices = np.argsort(
        prediction
    )[-top_k:][::-1]

    index_to_word = {
        index: word
        for word, index
        in tokenizer.word_index.items()
    }

    results = []

    for index in top_indices:

        word = index_to_word.get(
            int(index),
            "unknown"
        )

        confidence = float(
            prediction[index]
        )

        results.append(
            (word, confidence)
        )

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 LSTM Studio")

    st.markdown(
        "### Neural Text Generation"
    )

    st.divider()

    st.markdown("### Model")

    st.markdown(
        """
        **Architecture**

        `Embedding → LSTM → Dense`

        **Framework**

        TensorFlow / Keras

        **Task**

        Next-word prediction
        """
    )

    st.divider()

    st.markdown("### Model Stats")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Vocabulary",
            "8,978"
        )

    with col2:
        st.metric(
            "LSTM Units",
            "128"
        )

    st.metric(
        "Training Epochs",
        "10"
    )

    st.metric(
        "Max Sequence",
        str(max_len) if max_len else "745"
    )

    st.divider()

    st.caption(
        "Built with Python • TensorFlow • Keras • Streamlit"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="badge">
            ✦ DEEP LEARNING TEXT PREDICTOR
        </div>

        <h1>
            LSTM
            <span class="gradient-text">
                Word Studio
            </span>
        </h1>

        <p>
            Explore next-word prediction powered by a
            Long Short-Term Memory neural network trained
            on thousands of real-world quotes.
            Type a phrase, predict what comes next,
            and generate an entire continuation.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL STATUS
# ============================================================

if not model_loaded:

    st.error(
        """
        ⚠️ Model not found.

        Make sure these files are in the same folder as `app.py`:

        `lstm_model.keras`
        `tokenizer.pkl`
        `max_len.pkl`
        """
    )

    st.stop()


# ============================================================
# MAIN EDITOR
# ============================================================

st.markdown(
    '<div class="card-title">✍️ Text Playground</div>',
    unsafe_allow_html=True
)

seed_text = st.text_area(
    "",
    value="the future of artificial",
    height=150,
    placeholder="Start typing something...",
    label_visibility="collapsed"
)


# ============================================================
# CONTROLS
# ============================================================

col1, col2, col3 = st.columns([1.4, 1.4, 1])

with col1:

    generate_words = st.slider(
        "Words to generate",
        min_value=1,
        max_value=30,
        value=10
    )

with col2:

    top_k = st.slider(
        "Prediction alternatives",
        min_value=3,
        max_value=10,
        value=5
    )

with col3:

    st.write("")

    predict_button = st.button(
        "✨ Predict Next Word"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if not seed_text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        predicted_word, confidence = predict_next_word(
            seed_text
        )

        if predicted_word:

            st.markdown(
                f"""
                <div class="prediction">

                    <div class="card-title">
                        NEXT WORD
                    </div>

                    <div class="predicted-word">
                        {predicted_word}
                    </div>

                    <div style="
                        color:#94a3b8;
                        margin-top:8px;
                    ">
                        Model confidence:
                        <strong>
                            {confidence * 100:.2f}%
                        </strong>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🔮 Alternative Predictions"
            )

            predictions = get_top_predictions(
                seed_text,
                top_k
            )

            for word, score in predictions:

                col_a, col_b = st.columns(
                    [1, 4]
                )

                with col_a:

                    st.markdown(
                        f"**{word}**"
                    )

                with col_b:

                    st.progress(
                        min(score, 1.0),
                        text=f"{score * 100:.2f}%"
                    )


# ============================================================
# GENERATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="card-title">🚀 Text Generation</div>',
    unsafe_allow_html=True
)

if st.button(
    "Generate Text Continuation",
    key="generate"
):

    if not seed_text.strip():

        st.warning(
            "Enter a starting phrase first."
        )

    else:

        with st.spinner(
            "LSTM is generating..."
        ):

            generated = generate_text(
                seed_text,
                generate_words
            )

        st.markdown(
            f"""
            <div class="output-box">

                {generated}

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("---")

st.markdown("## 🧠 How It Works")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="card">

        ### 01 — Tokenization

        Your text is converted into numerical
        tokens using the trained Keras tokenizer.

        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="card">

        ### 02 — LSTM

        The sequence is processed by an LSTM
        network that learns relationships between
        previous words.

        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="card">

        ### 03 — Prediction

        The Dense layer produces probabilities
        for the next word in the vocabulary.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.markdown("---")

st.markdown("## 🔬 Model Architecture")

st.code(
"""
Input Text
    ↓
Keras Tokenizer
    ↓
Sequence Padding
    ↓
Embedding Layer
    ↓
LSTM Layer — 128 Units
    ↓
Dense Layer
    ↓
Softmax Probabilities
    ↓
Predicted Next Word
""",
language="text"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        LSTM Word Studio · Deep Learning Portfolio Project<br>
        Built with TensorFlow + Keras + Streamlit

    </div>
    """,
    unsafe_allow_html=True
)