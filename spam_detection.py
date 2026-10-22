import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample email dataset
data = {
"message": [
"Congratulations! You won a free lottery ticket",
"Win money now, click this link",
"Get a free gift card today",
"You have won a cash prize",
"Claim your free reward now",
"Meeting is scheduled for tomorrow",
"Please send me the project report",
"Can we meet at 5 PM?",
"Your assignment is ready",
"Let's discuss the project"
],
"label": [
"spam", "spam", "spam", "spam", "spam",
"ham", "ham", "ham", "ham", "ham"
]
}

df = pd.DataFrame(data)

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Check accuracy
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("Email Spam Detection Project")
print("----------------------------")
print("Model Accuracy:", accuracy)

# Test new emails
emails = [
"Congratulations you won a free prize",
"Please attend the meeting tomorrow"
]

email_vector = vectorizer.transform(emails)
results = model.predict(email_vector)

for email, result in zip(emails, results):
    print("\nEmail:", email)
print("Prediction:", result)
