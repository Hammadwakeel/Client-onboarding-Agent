import logging
from client_agent import clienthandler_agent
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
import os
from werkzeug.utils import secure_filename
import json
from pymongo import MongoClient
import urllib.parse
from openai import OpenAI


# Set up logging
log_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app.log')
logging.basicConfig(
    level=logging.INFO,  # Log to both terminal and file
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # This will output logs to the terminal
    ]
)

# Log application start
logging.info("Starting Flask application...")

openai_api_key = os.environ.get('OPENAI_API_KEY')
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    raise Exception("OPENAI_API_KEY environment variable not set.")
client = OpenAI()
from langchain_openai import OpenAIEmbeddings, OpenAI
# Get current working directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Paths to your resources
SERVICE_ACCOUNT_FILE = os.path.join(base_dir, 'chatbot-aia-435716-4e0ae920d93a.json')
pdf_path = os.path.join(base_dir, 'Autonomous_Intelligence_Overview.pdf')
save_faiss_path = os.path.join(base_dir, 'company_faiss_index')

# Define the scope for Google Calendar API
SCOPES = ['https://www.googleapis.com/auth/calendar']

# Email of the user to impersonate
DELEGATED_USER_EMAIL = 'laurent@autonomous-intelligence.fr'
sending_email_address = "laurent@autonomous-intelligence.fr"
sending_email_password = "gpem mqjx noml mtvj"
receiving_email = 'laurent@autonomous-intelligence.fr'
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = OpenAI()

# Your original credentials
username = 'Hasnain'
password = 'D6H%em$4De9t98G'

# URL-encode the username and password
encoded_username = urllib.parse.quote_plus(username)
encoded_password = urllib.parse.quote_plus(password)

# Construct the URI with encoded credentials
uri = f'mongodb+srv://{encoded_username}:{encoded_password}@serverlessinstance0.fqy1klf.mongodb.net/?retryWrites=true&w=majority&appName=ServerlessInstance0'

# Create the MongoDB client
client = MongoClient(uri)
db = client["autonomous_agent"]

agent = clienthandler_agent(sending_email_address, sending_email_password, receiving_email, llm, client, embeddings, pdf_path, save_faiss_path, SERVICE_ACCOUNT_FILE, SCOPES, DELEGATED_USER_EMAIL, db)

# Set up the Flask app
app = Flask(__name__)

# Define where to temporarily save uploaded files
UPLOAD_FOLDER = os.path.join(base_dir, 'tmp', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Limit upload size (optional)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']

        if file.filename == '':
            logging.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(file_path)
            logging.info(f"File {filename} saved at {file_path}")

            agent.update_db(file_path)
            logging.info(f"FAISS index updated with {filename}")

            return jsonify({"message": f"FAISS index updated with {filename}"}), 200
        else:
            logging.warning("Invalid file type")
            return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400
    except Exception as e:
        logging.error(f"Error during file upload: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        # Get the incoming message and sender's number
        incoming_msg = request.values.get('Body', '').lower()
        from_number = request.values.get('From', '')

        # Log the incoming message and sender's number
        logging.info(f"Message received from {from_number}: {incoming_msg}")

        # Manage session and get the state, answer, and chat history from the agent
        from_number_verified = agent.manage_session(from_number)
        answer, state, chat_history = agent.run(incoming_msg, from_number_verified)

        # Log the answer, state, and chat history
        logging.info(f"State for {from_number}: {state}")
        logging.info(f"Answer sent to {from_number}: {answer}")
        logging.info(f"Chat history with {from_number}: {chat_history}")

        # Prepare the response for Twilio
        response = MessagingResponse()
        response.message(answer)

        # Log that the response has been sent
        logging.info(f"Response sent to {from_number}")

        # Return the response
        return str(response)

    except Exception as e:
        # Log any error that occurs during the process
        logging.error(f"Error in webhook processing: {str(e)}")
        return str(e), 500


if __name__ == "__main__":
    try:
        port = int(os.environ.get('PORT', 8080))
        logging.info(f"Application running on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logging.error(f"Error starting application: {str(e)}")