# 🤖 AI Agent – Client Automation via WhatsApp + OpenAI

AI Agent is an intelligent, automated system that manages **client onboarding** via **WhatsApp** by leveraging **OpenAI's GPT-4o**, **LangChain**, **MongoDB**, **FAISS**, and **Google Calendar API**.

This project is built for companies looking to **automate their initial client interactions** — from gathering project requirements to finalizing meeting times, all done by a conversational agent.

---

## 📌 Features

- 💬 Conversational AI powered by OpenAI (GPT-4o / GPT-4 Turbo)
- 🧠 Session-aware chat using LangChain's memory
- 📄 PDF ingestion using FAISS for context-aware responses
- 🗓️ Google Calendar meeting scheduling automation
- 🔐 MongoDB-based session management and data storage
- 📞 WhatsApp integration using Twilio API
- ⚙️ Modular design for easy extension

---

## 🚀 Project Purpose

> The **main goal** of this project is to take project-related information from clients and **automatically arrange a meeting** with the company representative. This project is a core component of the company’s **client onboarding automation system**.

---

## 📂 Project Structure

```bash
ai-agent/
├── app.py                      # Flask app for Twilio webhook and file uploads
├── client_agent.py             # Agent orchestration logic (state, session, conversation)
├── requirements.txt            # Python dependencies
├── tmp/uploads/               # Folder for uploaded PDFs
````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Hammadwakeel/Client-onboarding-Agent.git
cd Client-onboarding-Agent
```

### 2. Install Dependencies

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Set up the following environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export PORT=8080  # Or any other port
```

You also need:

* A **Twilio WhatsApp Business account**
* A **MongoDB Atlas URI**
* A **Google Cloud Service Account** for Calendar API access
* A **PDF document** for FAISS indexing

### 4. Run Flask App (Twilio WhatsApp Integration)

```bash
python app.py
```

### 5. Run FastAPI App (Conversation Management)

```bash
uvicorn fastapi_conversation:app --host 0.0.0.0 --port 8000
```

---

## 🔁 WhatsApp Workflow

1. **User sends a message** via WhatsApp
2. Flask webhook receives it and triggers the `clienthandler_agent`
3. `clienthandler_agent`:

   * Checks chat state: greeting, gather\_info, arrange\_meeting
   * Chooses appropriate LLM prompt
   * Uses LangChain memory for context
   * Returns intelligent response
4. Agent stores conversation and chat state in MongoDB
5. If required, schedules a meeting using Google Calendar API

---

## 📥 PDF Upload Endpoint

```http
POST /upload_pdf
Content-Type: multipart/form-data
Form Field: file=<your_pdf_file.pdf>
```

This updates the FAISS vector index for referencing in client conversations.

---

## 📬 FastAPI Endpoints (Conversation Logic)

| Endpoint        | Method | Description                                       |
| --------------- | ------ | ------------------------------------------------- |
| `/conversation` | POST   | Accepts user questions and returns agent response |
| `/history`      | GET    | Returns current chat history                      |

---

## 🧠 AI Logic (Simplified State Machine)

| State             | Function            | Description                              |
| ----------------- | ------------------- | ---------------------------------------- |
| `greeting`        | `Greeting()`        | Initial hello and small talk             |
| `gather_info`     | `Gather_info()`     | Ask for project needs                    |
| `arrange_meeting` | `arrange_meeting()` | Confirm schedule and meeting preferences |
| `ending`          | (None)              | Graceful exit from conversation          |

---

## 📌 Tech Stack

* **Frameworks**: Flask, FastAPI
* **AI Models**: OpenAI GPT-4o, GPT-4 Turbo
* **Memory**: LangChain `ConversationTokenBufferMemory`
* **Database**: MongoDB Atlas
* **Vector Search**: FAISS
* **Auth/Calendar**: Google API
* **Messaging**: Twilio WhatsApp API

---

## ✅ Sample .env (Optional)

```
OPENAI_API_KEY=sk-xxxxxx
PORT=8080
MONGO_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net/
GOOGLE_SERVICE_ACCOUNT_PATH=chatbot-aia-435716-xxxxxx.json
```

---

## 📈 Future Improvements

* 🔄 Add support for rescheduling meetings
* 🗣️ Enable voice input using Web Speech API
* 🌐 Web dashboard for monitoring client sessions
* 🧩 Agent chaining with multiple specialized agents

---

