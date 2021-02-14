# Link between model.py and index.html to take user inputs and display results
# App for inference, will be packaged with Docker and deployed with Kubernetes
import flask
from flask import request
from model import predict_sentiment
import os

# Initialize the app
app = flask.Flask(__name__)


@app.route('/')
def index():

    # Displays the shown string above the user entered text
    header_review = "Review:"

    # Displays the show string above the model determined sentiment
    header_sentiment = "Sentiment:"

    print(request.args)

    # Contains a dictionary containing the parsed contents of the query string
    if(request.args):

        # Passes contents of query string to the prediction function contained in model.py
        x_input, prediction = predict_sentiment(request.args['text_in'])
        print(prediction[0]['prob'])

        # Indexes the returned dictionary for the sentiment probability
        if((prediction[0]['prob']) > 0.9):
            prediction = "Extremely Positive"
            return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_review=header_review, header_sentiment=header_sentiment)
        elif((prediction[0]['prob']) > 0.6 and (prediction[0]['prob']) <= 0.9):
            prediction = "Positive"
            return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_review=header_review, header_sentiment=header_sentiment)
        elif((prediction[0]['prob']) > 0.5 and (prediction[0]['prob']) <= 0.6):
            prediction = "Somewhat Positive"
            return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_review=header_review, header_sentiment=header_sentiment)
        elif((prediction[0]['prob']) > 0.4 and (prediction[0]['prob']) <= 0.5):
            prediction = "Somewhat Negative"
            return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_review=header_review, header_sentiment=header_sentiment)
        elif((prediction[0]['prob']) > 0.1 and (prediction[0]['prob']) <= 0.4):
            prediction = "Negative"
            return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_review=header_review, header_sentiment=header_sentiment)
        else:
            prediction = "Extremely Negative"
            return flask.render_template('index.html', text_in=x_input, prediction=prediction, header_review=header_review, header_sentiment=header_sentiment)

    # If the parsed query string does not contain anything then return index page
    else:
        return flask.render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
