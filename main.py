import os
import base64
import json
import pickle
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from google.auth.transport.requests import Request as GoogleRequest
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google import genai

load_dotenv()

app = FastAPI(title="Autonomous Email-to-Calendar Agent")

# Configurations from environment variables
PORT = int(os.getenv("PORT", 8000))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kathmandu")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Google API Scopes for reading emails and managing calendar events
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar'
]

def get_google_services():
    """Authenticates and returns Gmail and Calendar service instances using OAuth."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
        else:
            if not os.path.exists('credentials.json'):
                raise HTTPException(
                    status_code=500, 
                    detail="credentials.json file missing! Download your OAuth Client JSON from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
            
    gmail_service = build('gmail', 'v1', credentials=creds)
    calendar_service = build('calendar', 'v3', credentials=creds)
    return gmail_service, calendar_service

def parse_email_with_ai(email_content: str):
    """Uses Google Gemini to extract structured meeting parameters from email text."""
    prompt = f"""
    Analyze the following email content. Determine if it proposes a meeting, appointment, interview, or call.
    If yes, extract the title, start time (ISO 8601 format, e.g., 2026-11-05T15:00:00), end time (default to 30 mins after start if not specified), and attendee emails.
    Current Year Context: 2026.
    
    Email Content:
    "{email_content}"
    
    Return ONLY a valid JSON object in this exact schema (no markdown formatting, no backticks):
    {{
      "is_meeting": true/false,
      "title": "Meeting Title or Catchup",
      "start_time": "YYYY-MM-DDTHH:MM:SS",
      "end_time": "YYYY-MM-DDTHH:MM:SS",
      "attendees": ["email1@example.com"]
    }}
    """
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"LLM Parsing Error: {e}")
        return {"is_meeting": False}

@app.post("/webhook/gmail")
async def gmail_webhook(request: Request):
    """Webhook endpoint triggered automatically when new emails land in the inbox."""
    try:
        body = await request.json()
        message_data = body.get("message", {}).get("data", "")
        
        if not message_data:
            return {"status": "ignored, no pub/sub data"}
            
        gmail_service, calendar_service = get_google_services()
        
        # 1. Fetch the latest unread email
        results = gmail_service.users().messages().list(
            userId='me', q='is:unread label:inbox', maxResults=1
        ).execute()
        messages = results.get('messages', [])
        
        if not messages:
            return {"status": "no unread messages to process"}
            
        msg_id = messages[0]['id']
        msg = gmail_service.users().messages().get(userId='me', id=msg_id).execute()
        snippet = msg.get('snippet', '')
        
        # Get sender email address from message headers
        headers = msg.get('payload', {}).get('headers', [])
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
        
        # 2. Analyze via Gemini AI
        extracted = parse_email_with_ai(snippet)
        
        if extracted.get("is_meeting"):
            attendees_list = [{'email': sender}]
            for att in extracted.get("attendees", []):
                attendees_list.append({'email': att})
                
            event_body = {
                'summary': extracted.get('title', 'Synced Meeting'),
                'description': f"Auto-scheduled by Autonomous AI Agent from email snippet: {snippet}",
                'start': {'dateTime': extracted['start_time'], 'timeZone': TIMEZONE},
                'end': {'dateTime': extracted['end_time'], 'timeZone': TIMEZONE},
                'attendees': attendees_list,
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 1440},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }
            
            # 3. Push event straight into Google Calendar
            calendar_service.events().insert(calendarId='primary', body=event_body, sendUpdates='all').execute()
            
            # 4. Mark email as read so it isn't processed twice
            gmail_service.users().messages().batchModify(
                userId='me',
                body={'ids': [msg_id], 'removeLabelIds': ['UNREAD']}
            ).execute()
            
            return {"status": "success", "event_created": extracted.get('title')}
            
        return {"status": "no meeting intent found in latest email"}
        
    except Exception as e:
        print(f"Webhook Execution Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
