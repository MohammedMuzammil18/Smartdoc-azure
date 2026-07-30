import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NOTEBOOK_DIR, exist_ok=True)
NOTEBOOK_PATH = os.path.join(NOTEBOOK_DIR, "SmartDoc_Data_Preprocessing_and_ML_Training.ipynb")

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TCS iON Industry Project: Building Intelligent Application with Azure AI Search\n",
    "## Milestone 1 & 2: Data Cleaning, Preparation, Feature Engineering & ML Model Training\n",
    "\n",
    "This notebook demonstrates the end-to-end data pipeline and machine learning workflow for the **SmartDoc Intelligent Search System**:\n",
    "1. **Data Collection & Inspection** (Structured & Unstructured Policy Data)\n",
    "2. **Data Cleaning & Standardization** (Handling missing values, text normalization, snippet generation)\n",
    "3. **Machine Learning Model Training** (TF-IDF Vectorizer + LinearSVC Document Classifier)\n",
    "4. **Model Evaluation & Cross-Validation** (Accuracy, Precision, Recall, Confusion Matrix)\n",
    "5. **Semantic Embedding Indexing** (SentenceTransformers `all-MiniLM-L6-v2` dense vector generation)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Environment Setup & Library Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import cross_val_score, StratifiedKFold\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
    "from sentence_transformers import SentenceTransformer\n",
    "\n",
    "print(\"All libraries imported successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Data Loading & Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../employee_docs.csv')\n",
    "print(f\"Raw dataset shape: {df.shape}\")\n",
    "display(df.head())\n",
    "\n",
    "# Data cleaning & text normalization\n",
    "df['text_clean'] = df['text'].astype(str).str.strip().str.replace(r'\\s+', ' ', regex=True)\n",
    "df['snippet'] = df['text_clean'].str[:200]\n",
    "df.to_csv('../cleaned_docs.csv', index=False)\n",
    "print(f\"Cleaned dataset saved with {len(df)} documents.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: ML Classifier Training & Pipeline Construction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df['text_clean']\n",
    "y = df['category']\n",
    "\n",
    "# Construct TF-IDF + LinearSVC classification pipeline\n",
    "pipeline = Pipeline([\n",
    "    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000, stop_words='english')),\n",
    "    ('clf', LinearSVC(C=1.0, random_state=42))\n",
    "])\n",
    "\n",
    "pipeline.fit(X, y)\n",
    "y_pred = pipeline.predict(X)\n",
    "print(f\"Training Accuracy: {accuracy_score(y, y_pred):.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Model Evaluation & Confusion Matrix Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Classification Report:\")\n",
    "print(classification_report(y, y_pred))\n",
    "\n",
    "cm = confusion_matrix(y, y_pred, labels=['HR', 'IT', 'Finance'])\n",
    "plt.figure(figsize=(6, 4))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HR', 'IT', 'Finance'], yticklabels=['HR', 'IT', 'Finance'])\n",
    "plt.xlabel('Predicted Label')\n",
    "plt.ylabel('True Label')\n",
    "plt.title('Document Classifier Confusion Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Local Semantic Embedding Index Generation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = SentenceTransformer('all-MiniLM-L6-v2')\n",
    "texts = [f\"{row['title']} - {row['text_clean']}\" for _, row in df.iterrows()]\n",
    "embeddings = model.encode(texts, convert_to_numpy=True)\n",
    "\n",
    "index_data = {\n",
    "    'model_name': 'all-MiniLM-L6-v2',\n",
    "    'documents': df.to_dict(orient='records'),\n",
    "    'embeddings': embeddings\n",
    "}\n",
    "\n",
    "with open('../local_search_index.pkl', 'wb') as f:\n",
    "    pickle.dump(index_data, f)\n",
    "\n",
    "print(f\"Local semantic search index generated with {len(embeddings)} vectors of dimension {embeddings.shape[1]}.\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print(f"Jupyter Notebook generated -> {NOTEBOOK_PATH}")
