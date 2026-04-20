import sys
# --- Import Required Libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load the dataset
df = pd.read_excel(sys.argv[1])  # Update path if needed

# 2️⃣ Extract text and labels
texts = df.iloc[:, 0].astype(str).tolist()
original_labels = df.iloc[:, 2].astype(str).tolist()

# 3️⃣ Combine 'Partially Legitimate' and 'Synthetic' into one class
binary_labels = ['Legitimate' if label == 'Legitimate' else 'Synthetic' for label in original_labels]

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)

# 5️⃣ Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6️⃣ Define individual models
models = [
    ('svm', LinearSVC(random_state=42)),
    ('nb', MultinomialNB()),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# 7️⃣ Create Voting Classifier (majority voting)
voting_clf = VotingClassifier(estimators=models, voting='hard')

# 8️⃣ Train ensemble
voting_clf.fit(X_train_tfidf, y_train)

# 9️⃣ Predictions
y_pred = voting_clf.predict(X_test_tfidf)

# 🔟 Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", round(accuracy * 100, 2), "%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
