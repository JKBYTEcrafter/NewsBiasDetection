import os
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

class MLPipeline:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLPipeline, cls).__new__(cls)
            cls._instance.init_model()
        return cls._instance

    def init_model(self):
        """Load the ML artifacts securely"""
        # Load embedding model
        print("Loading embedding transformer...")
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Load logistic regression
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model", "lr_model.pkl"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weight file not found at {model_path}. Did you run train.py first?"
            )

        print("Loading logistic regression classifier...")
        self.classifier = joblib.load(model_path)

        # ── Compatibility patch for scikit-learn >= 1.5 ──────────────────────
        # Older pkl files stored 'multi_class' as an instance attribute.
        # scikit-learn 1.5+ removed it; patch it back so predict() works.
        if not hasattr(self.classifier, 'multi_class'):
            self.classifier.multi_class = 'auto'
            print("Applied multi_class compatibility patch (sklearn >= 1.5 detected).")

        # Patch individual estimators inside LogisticRegressionCV as well
        if hasattr(self.classifier, 'estimators_'):
            for est in self.classifier.estimators_:
                if not hasattr(est, 'multi_class'):
                    est.multi_class = 'auto'

        print("Model loaded successfully.")

    def predict(self, text: str):
        # 1. Embed exactly as trained
        embedding = self.embedder.encode([text])[0]

        # 2. Predict using LR weights
        reshaped_emb = np.array(embedding).reshape(1, -1)

        label_class = self.classifier.predict(reshaped_emb)[0]

        # 3. Get probabilities
        probabilities = self.classifier.predict_proba(reshaped_emb)[0]
        classes = self.classifier.classes_

        # Zip to dictionary map
        confidences = {str(k): float(v) for k, v in zip(classes, probabilities)}

        return {
            "prediction": str(label_class),
            "confidence_matrix": confidences
        }
