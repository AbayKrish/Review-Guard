from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from nltk.corpus import stopwords
import string
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np

# Load resources
cv = pickle.load(open("vector.pickel", "rb"))
le = pickle.load(open("labelEncoder.pickel", "rb"))
model = tf.keras.models.load_model("model.h5")

# Initialize Flask app
app = Flask(__name__)


# Function to clean text
def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

# Function to preprocess text
def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

# Function to stem words
def stem_words(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Function to lemmatize words
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Route for the home page
@app.route('/')
def home():
    return render_template("index.html")

# Route to process review
@app.route('/process_review', methods=['POST'])
def process_review():
    try:
        review_text = request.form['review_text']
        model_selected = request.form['model_select']

        # Clean and preprocess text
        text = clean_text(review_text)
        text = preprocess(text)
        text = text.lower()
        text = stem_words(text)
        text = lemmatize_words(text)

        # Transform text using CountVectorizer
        text = cv.transform([text])

        # Make prediction using the model
        y = model.predict(text)
        y_pred = np.argmax(y, axis=1)

        # Map prediction to label
        output = "Genuine" if y_pred else "Not genuine"
        
        return render_template("index.html", result=output, text=review_text)
    except Exception as e:
        error_message = "An error occurred: " + str(e)
        return render_template("index.html", error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
