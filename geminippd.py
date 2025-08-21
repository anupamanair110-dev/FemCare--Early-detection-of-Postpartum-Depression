import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import logging

app = Flask(__name__)
CORS(app)  

# generate API key using Google AI Studio and put here
genai.configure(api_key="AIzaSyBJY0VjuHqrlUMKAZslQFeyr0rgIEkNV5U")


generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 150,
    "stop_sequences": ["\"END\""],
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=(
        "You are a chatbot for Postpartum depression. The Postpartum Depression chatbot should respond with empathy and compassion, "
        "prioritizing the user's mental health. It must provide clear, non-judgmental information about postpartum depression, "
        "including symptoms, causes, and treatments, while recommending professional help when needed. The chatbot should encourage "
        "honest self-assessments, suggest self-care strategies, and offer supportive suggestions. In sensitive situations, it must "
        "guide users towards professional assistance, especially if serious symptoms arise. Privacy must be maintained, with reassurance "
        "given about data security. The chatbot should always handle the conversation with patience, motivation, and a reassuring tone, "
        "ensuring users feel comfortable seeking help. Always provide complete and coherent responses without leaving sentences unfinished. "
        "If the token limit is approaching, make the answer concise while ensuring key information is included. Prioritize important points, "
        "avoid abrupt endings, and organize content clearly using bullet points or lists if needed. If the response is summarized due to token limits, "
        "ensure it remains clear and suggest follow-up if necessary."
    ),
)

chat_session = model.start_chat()  # Start a chat session


@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Endpoint to handle messages from the user."""
    try:
        data = request.get_json()  
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        modified_message = f"{user_message} (answer within 150 tokens)"

        response = chat_session.send_message(modified_message)  # Get chatbot response

        # Parse the response as JSON and extract the relevant field
        cleaned_response = parse_response(response.text)

        return jsonify({"reply": cleaned_response})  # Return cleaned response as JSON

    except Exception as e:
        logging.error(f"Error in chatbot endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request"}), 500


def parse_response(response_text):
    """Ensure the response is returned as plain text for a normal chat format."""
    try:
        response_json = json.loads(response_text)

        if isinstance(response_json, list):
            response_json = response_json[0]  # If response is a list, get the first item

        # Extract response text from known keys or return as plain text
        return response_json.get("response") or response_json.get("reply") or response_json.get("message") or response_text

    except json.JSONDecodeError:
        return response_text  # If it's not JSON, return it as-is



@app.route("/greet", methods=["GET"])
def greet():
    """Endpoint to send an initial greeting message."""
    greeting_message = "Hello! I'm here to assist you with postpartum depression. How are you feeling today?"
    return jsonify({"reply": greeting_message})


if __name__ == "__main__":
    app.run(debug=True)