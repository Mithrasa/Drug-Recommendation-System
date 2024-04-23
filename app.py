import pickle
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd

app = Flask(__name__)
df = pd.read_csv("drugsComTrain_raw.tsv", sep="\t")

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Define stopwords and lemmatizer
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Function to preprocess and vectorize the input text
def preprocess_and_vectorize(text):
    # Preprocess text
    html_removed = BeautifulSoup(text, 'html.parser').get_text()
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', html_removed)
    lowercase_text = cleaned_text.lower().split()
    meaningful_words = [word for word in lowercase_text if word not in stop]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in meaningful_words]
    preprocessed_text = ' '.join(lemmatized_words)

    # TF-IDF vectorization
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])

    return vectorized_text

# Function to extract top drugs for a given condition
def top_drugs_extractor(condition):
    return df[(df['rating'] >= 9) & (df['usefulCount'] >= 90) & (df['condition'] == condition)] \
        .sort_values(by=['rating', 'usefulCount'], ascending=[False, False])['drugName'].head(7).tolist()

# Function to predict sentiment and print top drugs
def predict_and_print(text, label):
    target_conditions = {
        "Depression": "Depression",
        "Acne": "Acne",
        "Anxiety": "Anxiety",
        # "Pain": "Pain",
        "Birth Control": "Birth Control"
    }

    target = target_conditions.get(label, "Unknown Condition")

    if target != "Unknown Condition":
        top_drugs = top_drugs_extractor(target)
        top_drugs = list(set(top_drugs))

        print("text:", text, "\nCondition:", target)
        print("Top 3 Suggested Drugs:")
        if len(top_drugs) >= 3:
            print(top_drugs[0])
            print(top_drugs[1])
            print(top_drugs[2])
        else:
            print("Not enough drugs for the specified condition.")
    else:
        print("Can't recommend drugs when the condition is unknown.")
    print()


# Route for predicting sentiment
@app.route('/predict_sentiment',methods=['GET','POST'])
def predict_sentiment():
    # prediction =None
    if request.method == 'POST':
        # Get the input review text from the form
        review = request.form['review']
        
        # Preprocess and vectorize the input text
        preprocessed_and_vectorized_text = preprocess_and_vectorize(review)
        
        # Make prediction using the model
        prediction = model.predict(preprocessed_and_vectorized_text)[0]
        
        # Get top drugs for the predicted condition
        target_conditions = {
        "Depression": "Depression",
        "Acne": "Acne",
        "Anxiety":"Anxiety",
        # "Pain":"pain",
        "Birth Control": "Birth Control"
        }   
        target_condition = target_conditions.get(prediction, "Unknown Condition")
        top_drugs = top_drugs_extractor(target_condition)
    

        # Render HTML with prediction result and top drugs
        return render_template('index.html', prediction=prediction, top_drugs=top_drugs)
    else:
        return redirect(url_for("/"))
        

# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("blog.html")


    
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/home")
def home1():
    return redirect(url_for("/"))

if __name__ == '__main__':
  
    app.run(debug=True)
   
