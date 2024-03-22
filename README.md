# Sentiment Analysis Flash App

This repository contains a Flask app for sentiment analysis. The app analyzes sentiment of text reviews related to badminton.

## Folder Structure

- **Data/reviews_badminton**: Contains the data files used for sentiment analysis.
- **requirement.txt**: Contains a list of required Python packages. Install them using `pip install -r requirements.txt`.
- **app.py**: Flask app file. Run the app using `python3 app.py` in the terminal.

## Setup and Usage

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/sentiment-analysis.git
    ```

2. Navigate to the cloned repository directory:

    ```bash
    cd sentiment-analysis
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Flask app:

    ```bash
    python3 app.py
    ```

5. Open your web browser and go to `http://localhost:5000` to access the app.

## How to Use

1. Enter a review related to badminton in the text area provided.
2. Click on the "Predict Sentiment" button.
3. The app will analyze the sentiment of the review and display the result.

## About

This Flask app uses a machine learning model to perform sentiment analysis on text reviews. It is built for analyzing sentiment specifically in badminton-related reviews. The model is trained on a dataset of badminton-related reviews and predicts whether a review is positive, negative, or neutral.

## Contributors

- [Avinash](https://github.com/ashroyalc)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
