# Smart Game Review Analyzer

This project analyzes Steam reviews using multimodal Machine Learning models. It processes text, image, and tabular features to predict sentiment, authenticity, and visual match.

## Setup
pip install -r requirements.txt


## Run Order

1. Run image_scraper.ipynb to download images

2. Run M0_Data_Preparation.ipynb to Prepare data 

3. Train individual models
    - M1_CNN.ipynb
    - M2_LSTM.ipynb
    - M3_GRU.ipynb
    - M4_ANN.ipynb

4. Train fusion model
    - M5_Fusion.ipynb


5. Run comparison & ablation
    - comparison.py
    - ablation.py


6. Launch Streamlit demo
    - streamlit run app/app.py


## Expected outputs
- `saved_models/` — 5 .h5 model files
- `evaluation/figures/` — confusion matrices + training curves
- Streamlit app at http://localhost:8501

## Hardware
Tested on CPU-only. For faster training, use a GPU.

-----------------------------------------------------------------
## Test Cases

### Test 1 — Trusted Review

- Review: "One of the best grand strategy games. The new Infernals faction adds unique mechanics, the mid-game crisis keeps you engaged, and the UI improvements make managing a galactic empire far smoother than before."

- Playtime=120h, Score=9, Price=19.99, Votes=42
- Expect: 
Sentiment = "Positive"; Authenticity = "Authentic"; Trust class = 2 (TRUSTED); 

### Test 2 — Negative trusted

- Review: "Very disappointed, I've played 300 hours and this is the worst update. The new Infernals DLC broke several existing mechanics. After the patch, mid-game performance tanks and the AI empires make nonsensical decisions."
- Playtime=300, Score=3, Price=19.99, Votes=6
- Expect:
Sentiment = "Negative"; Authenticity = "Authentic"; Trust class = 2 (TRUSTED) or 1 (SUSPICIOUS)

### Test 3 — Suspicious Review

- Review: "Pretty fun. Would recommend." 
- Playtime=0.30, Score=8, Price=29.99, Votes=9
- Expect: 
Trust class = 1 (SUSPICIOUS)

### Test 4 — Fake Review

- Review: "Very disappointed. worst game eveeeeerrrrr!!!! it is west  of time."
- Playtime=0.00, Score=1, Price=0, Votes=1
- Expect:
Sentiment = "Negative"; Authenticity = "Fake"; Trust class = 0 (Fake) or 1 (SUSPICIOUS)
