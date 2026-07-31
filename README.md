# Multi disease prediction web app
## This web app work with Flask or streamlit by two diffrent files flask_app.py or st_app.py.

The app contains 9 models for predict 9 different diseases and when click predict button the output return in (confidence percentage / risk percentage / the prediction(low risk / high risk) / message from the indicator with predicts but in text / the type of disease(This is only in the model of anemia and say the type of anemia from 8 types)):

1_ Heart disease model:
 _ classification /Binary
 _ output: Low risk / High risk

2_ Anemia disease model:
 _ softmax (multi output)
 _ Healthy / types of anemia

3_ stroke disease model:
 _ classification / binary
 _ output: Low risk / High risk

4_ kidney disease model:
 _ softmax (multi output)
 _ output: Low risk / Moderate risk / High risk

5_ Diabetes prediction model:
 _ classification / binary

6_ Liver disease model:
 _ classification / binary

7_ Hypertension disease model:
 _ classification / binary

8_ chronic kidney disease model:
 _ classification / binary

9_ Alzheimer disease model:
 _ classification / binary

## AI Contribution in the development process
1_suggestion in training models process
2_Making the frontend folder for flask and contribute in making st_app.py ""Note :All logic From my developing and ai didn't contribute in the logic and the core of the app

| ##|     Model    | Test Accuracy | Test Recall | Test Loss |
|---|--------------|---------------|-------------|-----------|
| 1 | Heart Attack |    89.00%     |    94.12%   |   0.3593  |
| 2 |    Anemia    |    94.82%     |    92.23%   |   0.0837  |
| 3 |    Stroke    |    93.45%     |     85%     |   0.6410  |
| 4 |    Kidney    |    89.61%     |    89.61%   |   0.4046  |
| 5 |   Diabetes   |    92.19%     |    86.03%   |   0.1547  |
| 6 |    Liver     |    69.49%     |    68.89%   |   0.6089  |
| 7 | Hypertension |    91.96%     |    93.72%   |   0.2498  |
| 8 |Chronic Kidney|    96.50%     |    89.00%   |   0.0882  |
| 9 | Alzheimer's  |    89.77%     |    80.00%   |   0.4089  |
# Demo Url 
https://diseases-prediction-website-mosalah55.streamlit.app/

##License
See `LICENSE` file in the repo
## Screenshots from the app
<img width="1920" height="1020" alt="Screenshot 2026-07-31 161710" src="https://github.com/user-attachments/assets/4fc2f81a-7cf2-48f8-a7c6-6d65f8809c0a" />
<img width="1920" height="1020" alt="Screenshot 2026-07-31 161451" src="https://github.com/user-attachments/assets/f171b0cc-155b-4a32-8ced-be7e58da25b0" />
<img width="1920" height="1020" alt="Screenshot 2026-07-31 161413" src="https://github.com/user-attachments/assets/056b027e-13d1-4891-a11c-42597c7d57d5" />

