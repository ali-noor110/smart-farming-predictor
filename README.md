# Smart Farming Predictor
AI-powered crop recommendation system combining:
- **A\* Search** — state-space search with normalized Euclidean distance heuristic
- **Genetic Algorithm** — farming plan optimizer (200 generations, tournament selection)
- **Random Forest** — ML baseline classifier (2,200 samples, 22 crops, held-out accuracy reported at runtime)
## Project Structure
```
AI/
├── app.py                    # Streamlit UI (main entry point)
├── astar.py                  # A* Search algorithm
├── genetic_algorithm.py      # Genetic Algorithm optimizer
├── random_forest.py          # Random Forest trainer & predictor
├── test_project.py           # Focused smoke tests
├── data/
│   └── Crop_recommendation.csv # Kaggle dataset (offline)
├── random_forest_model.pkl   # Trained model (auto-generated)
└── requirements.txt
```
## Setup & Run
For Windows PowerShell:
```bash
 Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

python -m streamlit run app.py
```

Place the Kaggle `Crop_recommendation.csv` in `data/`. The app trains and caches
the Random Forest model locally on first run.

## Dataset
The source is the public Kaggle
[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
Dataset license and redistribution terms must be checked on Kaggle before public
distribution. The project does not require an account or network access after the
CSV is stored locally.

## Evaluation
The Random Forest uses a fixed 80/20 train-test split with seed 42 and
stratified sampling. The same 440 held-out records are used to evaluate A*.
The UI reports:

- Held-out accuracy for A* and Random Forest
- Runtime for A* and Random Forest
- A* nodes expanded
- Random Forest confusion matrix
- Top-K crop agreement
- GA best fitness by generation
- GA results across five fixed seeds
## Limitations
Recommendations are based on historical dataset patterns and textbook/learned
crop profiles. The application is a decision-support tool and is not a guarantee
of crop yield or a replacement for professional agronomic advice.