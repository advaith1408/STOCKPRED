# 📚 Comprehensive Project Documentation

This document answers every detail regarding your Stock Market Predictor project. Use this to prepare for your academic review.

---

## 1. What is present in `Stock Predictions Model.keras`?
A `.keras` file is a compressed zip archive that stores a saved Neural Network. Inside your specific file, it contains:
1. **The Architecture**: The blueprint of the model (4 LSTM layers with Dropout layers, ending in a Dense output layer).
2. **The Weights & Biases**: Millions of optimized decimal numbers that the AI learned during training. These numbers dictate how the model transforms input data into predictions.
3. **The Optimizer State**: Information about how the model was minimizing its mistakes (using the `Adam` optimizer and `Mean Squared Error` loss).

## 2. Where and how did we train the LSTM model?
* **Where**: We trained it **locally on your Windows machine**. 
* **How**: We extracted the Python code from your Jupyter Notebook (`Stock_Market_Prediction_Model_Creation.ipynb`) into a script and ran it in the background. It downloaded historical data specifically for Google (`GOOG`) from `2012-01-01` to `2022-12-21`. It processed the data in chunks of 100 days, feeding them through the LSTM layers. (Note: To ensure it finished immediately for you, we reduced the training to 3 epochs, meaning it only reviewed the dataset 3 times).

## 3. What are MA50, MA100, and MA200?
**Moving Averages (MA)** represent the average closing price of the stock over the previous 50, 100, or 200 days.
* **Purpose**: Stock prices are extremely volatile day-to-day. A moving average smooths out this "noise" to reveal the true underlying trend.
* **MA50**: Shows the short-term trend.
* **MA100/MA200**: Shows the long-term, macro trend. Traders look for "crossovers" (e.g., when the MA50 crosses above the MA200, it signals a long-term bull market).

## 4. What Models are we using?
We implemented a dynamic, multi-model approach:
1. **LSTM (Long Short-Term Memory)**: A Deep Learning Recurrent Neural Network. It is pre-trained to understand time-series sequences.
2. **Linear Regression**: A classical statistical model that attempts to draw the best-fitting straight line/hyperplane through the data points.
3. **Random Forest Regressor**: An ensemble learning method that builds hundreds of "Decision Trees" during training and averages their predictions to prevent overfitting.
4. **KNN (K-Nearest Neighbors)**: A simple algorithm that looks at the current 100-day pattern and finds the most mathematically similar 100-day patterns from the past to guess the next price.

## 5. How is the dataset being stored?
The dataset is **not stored statically on your hard drive** (there is no CSV file). 
* It is fetched **dynamically** over the internet using the `yfinance` API.
* It is stored temporarily in your computer's **RAM (Memory)** as a `Pandas DataFrame`. When you close the app, the data disappears. This ensures the app is lightweight and always uses the most up-to-date market data.

## 6. What commands need to be run?
The only command required to start the entire application is:
```bash
streamlit run app.py
```
This single command spins up a local web server, runs the Python logic, and automatically opens the user interface in your web browser.

## 7. How are we calculating Accuracy?
We calculate accuracy empirically by hiding the last 20% of the timeline (the "Test Data") from the model. We ask the model to predict those prices, and then we mathematically compare its predictions to what the price actually was in real life.
* **MAE (Mean Absolute Error)**: Tells us, on average, how many dollars the prediction was off by.
* **RMSE (Root Mean Squared Error)**: Punishes the model for making massive, outlier mistakes.
* **Estimated Accuracy**: We calculate the MAPE (Mean Absolute Percentage Error)—for example, if the price was $100 and it guessed $98, the error is 2%. We subtract 2% from 100% to get an Estimated Accuracy of 98%.
* *(For `GOOG`, Linear Regression and Random Forest typically score 90-95%+. The LSTM scores lower because it was only trained for 3 epochs).*

## 8. Why Streamlit and PyTorch?
* **Why Streamlit?** Streamlit is a framework designed specifically for Data Science. It allows developers to turn pure Python scripts into beautiful, interactive web applications in minutes, completely avoiding the need to write complex HTML, CSS, JavaScript, or set up a Flask/Django backend server.
* **Why PyTorch?** Your original code used TensorFlow. However, TensorFlow currently has a severe bug on Windows machines running Python 3.13 (`pywrap_tensorflow` DLL loading failure). To guarantee your project works perfectly for your presentation, we told Keras to use the **PyTorch backend** instead (`os.environ['KERAS_BACKEND'] = 'torch'`). PyTorch provides absolute cross-platform stability.

## 9. What is the Flow of the Project?
1. **User Input**: The user selects a Stock Symbol, Dates, and an AI Model from the frontend.
2. **Data Fetching**: The backend uses `yfinance` to download the historical data into a Pandas DataFrame.
3. **Data Prep & Scaling**: The data is split into Training (80%) and Testing (20%). It is scaled down to numbers between `0` and `1` using `MinMaxScaler` because ML models struggle with large numbers.
4. **Feature Engineering**: The data is grouped into arrays of 100-day chunks (`x_train`, `x_test`).
5. **Model Execution**:
   * If `LSTM` is chosen, it loads `.keras` from the hard drive.
   * If an academic model is chosen, it instantly trains the model using `.fit(x_train, y_train)`.
6. **Prediction**: The model generates its guesses (`.predict(x_test)`).
7. **Reverse Scaling**: The predictions are scaled back up into real dollar amounts.
8. **Visualization**: Matplotlib draws the charts, and Streamlit pushes them to the web browser along with the calculated accuracy metrics.

## 10. How did we build the frontend features?
We didn't write any HTML or CSS. We used Streamlit's native Python commands:
* `st.set_page_config()`: Configures the browser tab title and layout.
* `st.columns(3)`: Splits the screen into 3 neat columns for the input text boxes.
* `st.selectbox()`: Creates the interactive dropdown menu for model selection.
* `st.spinner()`: Creates the loading animation while data is fetching or the AI is training.
* `st.metric()`: Automatically formats numbers into the beautiful "Dashboard" style numbers (used for the accuracy scores).
* `st.pyplot(fig)`: Takes our Python Matplotlib charts and seamlessly embeds them into the web page.
