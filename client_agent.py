import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import datetime
from dateutil import parser
import uuid
import pytz
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.document_loaders import (PyPDFLoader)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


class clienthandler_agent:
    def __init__(self, username, password, receiving_email, llm, client, embeddings, pdf_path, save_faiss_path,
                 SERVICE_ACCOUNT_FILE, SCOPES, DELEGATED_USER_EMAIL, db):
        from openai import OpenAI

        self.client = client
        self.llm = llm
        self.embeddings = embeddings
        self.save_faiss_path = save_faiss_path
        self.retriver = self.create_db(pdf_path)
        self.SERVICE_ACCOUNT_FILE = SERVICE_ACCOUNT_FILE
        self.SCOPES = SCOPES
        self.DELEGATED_USER_EMAIL = DELEGATED_USER_EMAIL
        self.BRAZIL_TZ = pytz.timezone('America/Sao_Paulo')

        self.username = username
        self.password = password
        self.receiving_email = receiving_email

        # Initialize the session tracking variables

        self.agents_collection = db["agents_status"]
        self.chats_collection = db["chats"]

        self.send_email_tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_response_email",
                    "description": "send the email with the given subject and body in string. Call this method when data is in proper extracted and prepare the subject and also prepare the body and send the email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email_subject": {
                                "type": "string",
                                "description": "describe the subject of the email here in string",
                            },
                            "email_body": {
                                "type": "string",
                                "description": "describe the body of the email here in string with extracted information in json with some text explanation.",
                            }

                        },
                        "required": ["email_subject", 'email_body'],
                        "additionalProperties": False,
                    },
                }
            }
        ]
        self.meet_tools = [
            {
                "type": "function",
                "function": {
                    "name": "arrange_meeting",
                    "description": "Arrange the meeting with the given email. Call this whenever you need to arrange the meeting, and the email is given by the user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "User's email ID for arranging the meeting using Google Meet API",
                            },
                            "start_event_time": {
                                "type": "string",
                                "description": " provide the starting event time for arranging the meeting using Google Meet API",
                            },
                        },
                        "required": ["email", "start_event_time"],
                        "additionalProperties": False,
                    },
                }
            }
        ]

    def create_db(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        all_documents = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        document_collection = text_splitter.split_documents(all_documents)
        faiss_index = FAISS.from_documents(document_collection, self.embeddings)
        faiss_index.save_local(self.save_faiss_path)
        return faiss_index.as_retriever(search_kwargs={"k": 2})

    def update_db(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        all_documents = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        document_collection = text_splitter.split_documents(all_documents)
        faiss_index = FAISS.load_local(self.save_faiss_path, self.embeddings,
                                       allow_dangerous_deserialization=True)
        additional_index = FAISS.from_documents(document_collection, self.embeddings)
        faiss_index.merge_from(additional_index)
        faiss_index.save_local(self.save_faiss_path)
        self.retriver = faiss_index.as_retriever(search_kwargs={"k": 2})

    # Function to select state based on chat history
    def select_state(self, chat_history, current_status):
        output_format = '''
        Answer according to the following json format:
        Output Format:
        {
        "State": here you will select the one state based on chat history: "greeting", "introduction", "gather_info", "recommendation", "arrange_meeting"
        }'''

        prompt = f"""
        You are expert agent to switch the state smoothly and dont add any rough transition. I want full smoothness.
        Based on the below chat history, decide the state for the agent. The state can be greeting, introduction, gather information, or arrange a meeting.
        switch to greeting if greeting messages are not in chat history.
        switch to introduction state if greetings are successful executed in chathistory. you will move to the introduction.
        swtch to gather information state if introduction is done. You will smoothly switch to gather information state and introduction state should be short and decide according to users question but should be short question answer pairs not too much long and boring.
        swtch to the recommendation state if all the following questions are present in chat history other wise state will be gatehr info. Answer of each question does not too much matter.
        - First name 
        - Last name
        - Company name the person is working with.
        - Project information about what the person wants to do, what the expected result is, what technologies or frameworks are expected to be used, etc.
        - Project timing info when they need to start, when they want to finish, or if there's no timing info yet.
        - Additional context there is more context to the project such as if it's a university project, startup-related, government project, or anything else.
        - Budget if there is already a specific budget for the project.
        These questions is present in chat history then swtich to recommendation state
        switch to the arrange_meeting state when summary of what user want basically according to chat history and and why we are best for their project is done and performed clearly in chat history and also user have some more question are properly answer by model and present in chathistory.  These two thing are clearly mention after gather info summary and mention why we are best for their project if answer are not fully hundred percent delivered and anything is missing then dont move to next state and projects related question are fully answered how it will be successlet’s see how we can help you make a successful project ? and if everything frequent and smooth then move to the next arrange_meeting state.
        You will be in arrange meeting until user provide the email for arranging meeting.
        queue state:
        {current_status}
        this will show if question first is empty means insert "greeting" in state if greeting is alreading in queue thenfind the state next state that will be introduction and then gather_info, and then arrange_meeting abd then ending and dont move back. Keep moving formward

        Chat History:
        {chat_history}
        """ + output_format
        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Select the next state"}
            ]
        )
        json_data = json.loads(response.choices[0].message.content)
        return json_data['State']

    # Function to manage greeting
    def Greeting(self, question, chat_history, context):
        prompt = f"""
        Your name is Autonomous Intelligence Development.
        You are an expert agent to ask some initial greetings to users, 
        examples 
        say "Hello, thank you for contacting Autonomous Intelligence Development, we are pleased to meet you.
We are here to help you to develop and integrate AI" just like response not same but structure and interaction should be just like this.. In greeeting also consider the following context if important otherwise your response should be concise and to the point. Your task is to generate a concise response and your task is just greeting nothing else. if question irrelevant then try to avoid answer this type of question ask the user to the point

        context
        {context}

        Chat History:
        {chat_history}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}"}
            ]
        )
        return response.choices[0].message.content

    def introduction(self, question, chat_history, context):
        prompt = f"""
        Your name is Autonomous Intelligence Development.
        You are an expert agent to provide some introduction of company according to following context and you will introduce Autonomous Intelligence Development  and add some introduction in concise way and ask the user to say question find our company. dont add detail  and gave response in concise way and if question irrelevant then try to avoid answer this type of question ask the user to the point. Your response should be concise and introduce the company just.
        Answer structure:
        company introduction in concie and short way and then answer the question.

        Context:
        Autonomous Intelligence is a leading AI solutions provider, operating across Europe, South America, and Southeast Asia. Specializing in developing cutting-edge artificial intelligence applications, they focus on helping businesses integrate AI-driven technologies to enhance operations, productivity, and sustainability.
        Core Services:
        AI Application Development
        Process Automation
        Custom AI Models
        Generative AI and Predictive Systems
        Computer Vision
        Data Engineering
        Values and Vision
        Sustainability:
        Impact

        {context}
        Chat History:
        {chat_history}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}"}
            ]
        )
        return response.choices[0].message.content

    # Function to gather information
    def Gather_info(self, question, chat_history, context):
        prompt = f"""
        Your name is Autonomous Intelligence Development.
        You will also answer the user query and also consider the context if needed. Dont answer if there is irrelevant question.
        You are an information-gathering agent. Your task is to politely gather the following information from the user step by step and dont miss any question and try to engage the user to answer the question and make your answer concise to the point. If user question irrelevant then say the user to answer the provided question just and say any thing irrelevant thing. gather all information by asking question step by step and dont miss anything and answer should be concise. If user dont answer any question then move to the next question and i want all to ask all question. Ask following question:

        - First name: Ask the user's first name.
        - Last name: Ask the user's last name.
        - Company name: Gather the name of the company the person is working with.
        - Project information: Ask for details about what the person wants to do, what the expected result is, what technologies or frameworks are expected to be used, etc.
        - Project timing info: Ask if the project is urgent, when they need to start, when they want to finish, or if there's no timing info yet.
        - Additional context: Ask if there is more context to the project such as if it's a university project, startup-related, government project, or anything else.
        - Budget: Politely ask if there is already a specific budget for the project.

        Please ask these questions one by one and gather all required information while keeping the conversation friendly and engaging.
        Once you have asked all questions, please inform the user that you woud like to move to explain how we are able to help for the project or the questions, still in a friendly and engaging way. i.e. : "let’s see how we can help you make a successful project ?"
        Answer question from the provided context but also ask question that are given above
        Company Context:
        {context}
        Chat History:
        {chat_history}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}"}
            ]
        )
        return response.choices[0].message.content

    def recommendation(self, question, chat_history, context):
        prompt = f"""
        Your name is Autonomous Intelligence Development.
        You will also answer the user query and also consider the context if needed. Dont answer if there is irrelevant question.
        kindly consider this question to ask and answer this user wants let’s see how we can help you make a successful project ? if question is already in chathistory then dont need to ask this again and again.
        Kindly generate a concise and to the point response and dont need to explain irrelevant text and i want just concise response.
        You are an expert agent to provide the summary what type of project user want in fully concise and also add the response the proper answer in one line why we are best for their project and answer the user question to convince and gave reason why we are best for their project. If summary is already provide in chat history then dont need to provide the chat history again and again. Your task is to answer the user in concise way why we are best and dont need to explain irrelevant text and just to the point.
        Your answer should be related and concise.
        A the end, let's inform the user that we are moving to an appointment booking phase in a friendly and engaging way. i.e : "let’s now see if we can schedule a meeting with you to talk this project in more details ?"
        Context:
        Autonomous Intelligence is a leading AI solutions provider, operating across Europe, South America, and Southeast Asia. Specializing in developing cutting-edge artificial intelligence applications, they focus on helping businesses integrate AI-driven technologies to enhance operations, productivity, and sustainability.
        Core Services:
        AI Application Development
        Process Automation
        Custom AI Models
        Generative AI and Predictive Systems
        Computer Vision
        Data Engineering
        Values and Vision
        Sustainability:
        Impact

        {context}
        Chat History:
        {chat_history}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}"}
            ]
        )
        return response.choices[0].message.content

    def get_current_time_iso(self):
        # Define the time zone (example: 'America/Sao_Paulo' for -03:00 offset)
        timezone = pytz.timezone('America/Sao_Paulo')

        # Get the current time in the specified time zone
        now = datetime.datetime.now(timezone)

        # Convert to ISO 8601 format
        iso_format_time = now.isoformat()

        return iso_format_time

    def authenticate_google_api(self):
        """Authenticate using Service Account with Domain-Wide Delegation and return the Google Calendar API service."""
        credentials = service_account.Credentials.from_service_account_file(
            self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES
        )
        delegated_credentials = credentials.with_subject(self.DELEGATED_USER_EMAIL)
        service = build('calendar', 'v3', credentials=delegated_credentials)
        return service

    def list_google_meet_events(self, service, time_min=None):
        """List only Google Meet events on the user's primary calendar."""
        if time_min is None:
            time_min = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time

        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        google_meet_events = []

        if not events:
            return None
        else:
            for event in events:
                # Only consider events that have a Google Meet link
                if 'conferenceData' in event and 'entryPoints' in event['conferenceData']:
                    for entry_point in event['conferenceData']['entryPoints']:
                        if entry_point['entryPointType'] == 'video' and entry_point['uri'].startswith(
                                "https://meet.google.com/"):
                            google_meet_events.append(event)
                            break  # Exit after finding the Google Meet link to avoid duplicates

            # Return the end time of the last Google Meet event in UTC
            if google_meet_events:
                last_event = google_meet_events[-1]
                last_event_end = last_event['end'].get('dateTime', last_event['end'].get('date'))
                return last_event_end
            else:
                return None

    def create_next_meeting(self, email, service, last_event_end):

        """Create a new meeting immediately after the last Google Meet event."""
        try:

            # Parse the last_event_end time into a datetime object
            last_event_end_dt = parser.isoparse(last_event_end)
            next_meeting_start = last_event_end_dt
            next_meeting_end = next_meeting_start + datetime.timedelta(hours=1)

            # Convert to UTC for Google Calendar
            next_meeting_start_utc = next_meeting_start.astimezone(pytz.utc)
            next_meeting_end_utc = next_meeting_end.astimezone(pytz.utc)

            # Create the event with Brazil timezone (UTC-3)
            event = {
                'summary': 'Follow-up Meeting',
                'location': 'Online',
                'description': 'This is a follow-up meeting after the last Google Meet event.',
                'start': {
                    'dateTime': next_meeting_start_utc.isoformat(),
                    'timeZone': 'America/Sao_Paulo',
                },
                'end': {
                    'dateTime': next_meeting_end_utc.isoformat(),
                    'timeZone': 'America/Sao_Paulo',
                },
                'attendees': [
                    {'email': email},  # You can add more attendees
                ],
                'conferenceData': {
                    'createRequest': {
                        'conferenceSolutionKey': {
                            'type': 'hangoutsMeet'
                        },
                        'requestId': str(uuid.uuid4())  # Use a unique request ID
                    }
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 10},
                    ],
                },
            }

            event = service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1
            ).execute()

            # Get the end time from the event created
            next_meeting_end = parser.isoparse(event['end']['dateTime'])

            # Convert start and end times to Brazil time zone
            next_meeting_start_brazil = next_meeting_start_utc.astimezone(self.BRAZIL_TZ)
            next_meeting_end_brazil = next_meeting_end.astimezone(self.BRAZIL_TZ)

            return (
                f"""Follow-up meeting created.\n
Next meeting will start at: {next_meeting_start_brazil.strftime('%A, %B %d, %Y at %I:%M %p %Z')}\n
Next meeting will end at: {next_meeting_end_brazil.strftime('%A, %B %d, %Y at %I:%M %p %Z')}\n
Google Meet link: {event['conferenceData']['entryPoints'][0]['uri']}""",
                event['end']['dateTime']
            )

        except Exception as e:
            return (f"An error occurred while creating the meeting: {e}", None)

    def arrange_meeting(self, email, start_event_time):
        service = self.authenticate_google_api()

        # Step 2: Create the next meeting based on the last event end time
        if start_event_time:
            # Corrected the function call by adding the missing 'email' argument
            result, new_last_event_end = self.create_next_meeting(email, service, start_event_time)
            if new_last_event_end:
                start_event_time = new_last_event_end
            return result
        else:
            # If no events found, schedule a meeting starting now
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            last_event_end = now.isoformat()
            # Corrected the function call by adding the missing 'email' argument
            result, new_last_event_end = self.create_next_meeting(email, service, last_event_end)
            if new_last_event_end:
                last_event_end = new_last_event_end
            return result

    # Function to arrange a meeting
    def arrange_meeting_agent(self, phone_number, question, chat_history, context, last_event_end):
        prompt = f"""
        Your name is Autonomous Intelligence Development.
        Ask question start something like this let’s now see if we can schedule a meeting with you to talk this project in more details ? if question is already in chathistory then dont need to ask this again and again.
        You will also answer the user query and also consider the context if needed. Dont answer if there is irrelevant question.
        Kindly generate a concise and to the point response and dont need to explain irrelevant text and i want just concise response.
        Your task is to arrange the meeting ask the user to provide the email and select the date according to provided date of event in calender {last_event_end} and suggestion next future four dates based on time with time duration should be 30 minutes and 1 hours understand and it suggest futures dates means arrange meeting in one try and suggest timing and dates in future as compared to {last_event_end} and allow the user to select the time/date next to this last date {last_event_end} mean suggest futures dates from this date {last_event_end} and time and then arrange meeting and dont need to answer irrelevant question and you will just arrange the meeting and ask the user for the email and select the four different possibility of times and date. you task is ti finalizing the meeting arrangements. Ask the user email for the google meet. You are a helpful meeting arrangement assistant. Use the supplied tools to assist the user. Ask the user again and again until he provide the email for meeting arrangement and select the date for meeting if he dont want meeting ask him to take time and contact us again. I want to pass the start event time according to following format {last_event_end}. Kindly show the next four upcoming times according to this {last_event_end} with duration should be 1 to 2 hours and 3 should be with different with duration 2 and 3 hours  times and 1 should be with different date and pass the date according to this format {last_event_end} in tool and want to pass the start event time according to following format {last_event_end}.

        Answer question from the provided context but also ask question that are given above
        Company Context:
        {context}
            Chat History:
            {chat_history}
            """
        # Define the messages
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"late date of event in calender {last_event_end}   Here Question: {question}"}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",  # Assuming gpt-4 is the correct model name
            messages=messages,
            tools=self.meet_tools,
        )
        if response.choices[0].message.content:
            return response.choices[0].message.content, False

        else:
            tool_call = response.choices[0].message.tool_calls[0]
            # Extract the function name and arguments
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Call the arrange_meeting function with extracted arguments
            if function_name == "arrange_meeting":
                print(arguments["start_event_time"])
                result = self.arrange_meeting(arguments['email'], arguments["start_event_time"])
                body = f'''
                here is following chat history of client:
                {chat_history}
                Human: {question}
                Ai: {result}'''

                self.send_response_email(f"Chat History of client {phone_number}", body)
                return result, True
        return None

    def send_response_email(self, email_subject, email_text):

        # Email settings
        sender_email = self.username
        receiver_email = self.receiving_email
        password = self.password
        smtp_server = "smtp.gmail.com"
        port = 587

        message = MIMEMultipart("alternative")
        message["Subject"] = email_subject
        message["From"] = sender_email
        message["To"] = receiver_email

        text = email_text

        part1 = MIMEText(text, "plain")
        message.attach(part1)

        try:
            server = smtplib.SMTP(smtp_server, port)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            server.quit()
            return "Successfully sent email"
        except Exception as e:
            return f"Error: {e}"

    def send_email_agent(self, chat_history, phone_number):
        prompt = """
       extract this information from chat history send this extracted as a email body and set the suitable title with proper given phone number given in question  and send the email just set subject "extract information from phone number(replace with acutal phone number)" or also use the name. Making suject and body for email in such a way it should not be detected as spam. Dont need to mention irrelevant thing in email body and add the first sentence at inital greeting and then show the json data and dont neeed to ask that reply us if you want some query etc question at the end and just end the email message body with some good words.

            {
            "first_name": "Extract the user's first name.",
            "last_name": "Extract the user's last name.",
            "company_name": "Extract the name of the company the person is working with.",
            "project_information": "Extract details about what the person wants to do, what the expected result is, what technologies or frameworks are expected to be used, etc.",
            "project_timing_info": "Extract if the project is urgent, when they need to start, when they want to finish, or if there's no timing info yet.",
            "additional_context": "Extract if there is more context to the project such as if it's a university project, startup-related, government project, or anything else.",
            "budget": "Extract if there is mentioned a specific budget for the project.", 
            }
            """
        # Define the messages
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"here is provided phone number: {phone_number}, chat hstory: {chat_history}"}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",  # Assuming gpt-4 is the correct model name
            messages=messages,
            tools=self.send_email_tools,
        )
        if response.choices[0].message.content:
            return False
        else:
            tool_call = response.choices[0].message.tool_calls[0]
            # Extract the function name and arguments
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Call the arrange_meeting function with extracted arguments
            if function_name == "send_response_email":
                result = self.send_response_email(arguments['email_subject'], arguments['email_body'])

            return True

    def create_chat(self, phone_number):
        self.chats_collection.insert_one({
            "_id": phone_number,
            "history": []
        })

    def update_chat(self, phone_number, question, answer):
        self.chats_collection.update_one(
            {"_id": phone_number},
            {"$push": {"history": {"question": question, "response": answer}}}
        )

    def load_chat(self, phone_number):
        chat_record = self.chats_collection.find_one({"_id": phone_number})
        if not chat_record:
            raise Exception(f"Phone number {phone_number} not found.")
        return chat_record.get('history', [])

    def get_state_data(self, phone_number):
        state_data = self.agents_collection.find_one({"_id": phone_number})
        if not state_data:
            raise Exception(f"Phone number {phone_number} not found.")
        return state_data

    def update_state_data(self, phone_number, end, state_queue, send_status):
        self.agents_collection.update_one(
            {"_id": phone_number},
            {"$set": {
                "end": end,
                "state_queue": state_queue,
                "send_status": send_status
            }},
            upsert=True
        )

    # Main flow manager
    def run(self, question, phone_number):

        # Load chat history
        db_chat_history = self.load_chat(phone_number)

        # Load state data from MongoDB
        state_data = self.get_state_data(phone_number)
        end = state_data['end']
        state_queue = state_data['state_queue']
        send_status = state_data['send_status']

        memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=8000)

        # Load the previous conversation into memory
        for entry in db_chat_history:
            memory.save_context({"input": entry["question"]}, {"output": entry["response"]})

        # Load chat history from memory
        chat_history = memory.load_memory_variables({}).get('history', [])

        # If conversation ended, return message
        if end:
            return "Conversation ended.", self.load_chat(phone_number), None

        # Get the current state
        state = self.select_state(chat_history, state_queue)

        # Handle the arrange_meeting state
        if state == "recommendation" and "recommendation" not in state_queue:
            while not send_status:
                send_status_value = self.send_email_agent(chat_history, phone_number)
                self.update_state_data(phone_number, end, state_queue, send_status_value)
                # Load state data from MongoDB
                state_data = self.get_state_data(phone_number)
                end = state_data['end']
                state_queue = state_data['state_queue']
                send_status = state_data['send_status']

        # Append the current state to the state queue if not already present
        if state not in state_queue:
            state_queue.append(state)
            self.update_state_data(phone_number, end, state_queue, send_status)

        # Handle different states and generate responses
        if state == "greeting":
            context = self.retriver.invoke('question')
            response = self.Greeting(question, chat_history, context)
        elif state == "introduction":

            context = self.retriver.invoke('question')

            response = self.introduction(question, chat_history, context)
        elif state == "gather_info":
            context = self.retriver.invoke('question')
            response = self.Gather_info(question, chat_history, context)
        elif state == "recommendation":
            context = self.retriver.invoke('question')
            response = self.recommendation(question, chat_history, context)
        elif state == "arrange_meeting":
            service = self.authenticate_google_api()

            # Step 1: List upcoming Google Meet events and get the end time of the last one
            last_event_end = self.list_google_meet_events(service)

            context = self.retriver.invoke('question')
            if last_event_end:
                response, end_value = self.arrange_meeting_agent(phone_number, question, chat_history, context,
                                                                 self.get_current_time_iso())

            else:
                response, end_value = self.arrange_meeting_agent(phone_number, question, chat_history, context,
                                                                 self.get_current_time_iso())

            self.update_state_data(phone_number, end_value, state_queue, send_status)
        else:
            response = "Conversation ended."

        if response != "Conversation ended.":
            self.update_chat(phone_number, question, response)

        return response, self.load_chat(phone_number), state

    def manage_session(self, phone_number):
        if phone_number in self.agents_collection.distinct('_id'):
            return phone_number

        # Initialize session in MongoDB
        self.create_chat(phone_number)
        self.update_state_data(phone_number, False, [], False)
        return phone_number