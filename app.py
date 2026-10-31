import streamlit as st
import os
import json
import base64
from datetime import datetime, timedelta
from urllib.parse import urlencode, urlparse, parse_qs
from dotenv import load_dotenv

# Google libraries
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# AI
import google.generativeai as genai

# Database
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

# ============ DATABASE SETUP ============
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///./data.db"))
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class User(Base):
    __tablename__ = "users"
    email = Column(String, primary_key=True)
    access_token = Column(String)
    refresh_token = Column(String)
    auto_scan = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ProcessedEmail(Base):
    __tablename__ = "processed_emails"
    email_id = Column(String, primary_key=True)
    user_email = Column(String)
    processed_at = Column(DateTime, default=datetime.utcnow)

class CalendarEvent(Base):
    __tablename__ = "calendar_events"
    id = Column(String, primary_key=True)
    user_email = Column(String)
    title = Column(String)
    date = Column(String)
    time = Column(String)
    meeting_link = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ============ CONFIG ============
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8501")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 1.5 Flash with enforced JSON output
ai_model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config={"response_mime_type": "application/json"}
)

CLIENT_CONFIG = {
    "web": {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [REDIRECT_URI]
    }
}

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/userinfo.email'
]

# ============ HELPER FUNCTIONS ============

def get_oauth_flow():
    return Flow.from_client_config(
        CLIENT_CONFIG, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )

def refresh_credentials_if_needed(credentials):
    """Auto-refresh expired access tokens using refresh token"""
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        # Save new token to DB
        with SessionLocal() as db:
            user = db.query(User).filter(User.email == st.session_state.user_email).first()
            if user:
                user.access_token = credentials.token
                db.commit()
        st.session_state.credentials = credentials
    return credentials

def get_email_body(message):
    """Properly decode the full email body from Gmail's base64 format"""
    payload = message.get('payload', {})
    
    # Simple message
    if payload.get('mimeType') == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        if data:
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    
    # Multipart message - find the text/plain part
    parts = payload.get('parts', [])
    for part in parts:
        mime = part.get('mimeType', '')
        if mime == 'text/plain':
            data = part.get('body', {}).get('data', '')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        # Handle nested multipart (common in Gmail)
        if mime.startswith('multipart/'):
            for subpart in part.get('parts', []):
                if subpart.get('mimeType') == 'text/plain':
                    data = subpart.get('body', {}).get('data', '')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    
    # Fallback to snippet
    return message.get('snippet', '')

def analyze_email_with_ai(email_content):
    """Use Gemini's structured JSON mode for reliable parsing"""
    prompt = f"""Analyze this email and extract meeting information.

Email:
{email_content[:3000]}

Return a JSON object with these exact fields:
{{
  "is_meeting": boolean,
  "title": "string or null",
  "date": "YYYY-MM-DD or null",
  "time": "HH:MM or null (24-hour format)",
  "timezone": "IANA timezone like America/New_York or null",
  "meeting_link": "URL or null",
  "importance": "high/medium/low"
}}

If no meeting is found, return: {{"is_meeting": false}}
"""
    
    try:
        response = ai_model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"AI parsing error: {e}")
        return {"is_meeting": False}

def parse_meeting_datetime(date_str, time_str, duration_hours=1):
    """Safely parse date/time and handle day rollover (e.g., 23:00 + 1hr)"""
    try:
        start_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(hours=duration_hours)
        return start_dt, end_dt
    except ValueError as e:
        raise ValueError(f"Invalid date/time format: {date_str} {time_str} - {e}")

def create_calendar_event(service, meeting_data):
    """Create calendar event with proper datetime handling"""
    start_dt, end_dt = parse_meeting_datetime(
        meeting_data['date'], 
        meeting_data['time']
    )
    
    timezone = meeting_data.get('timezone') or 'UTC'
    
    event = {
        'summary': meeting_data.get('title', 'Meeting'),
        'start': {
            'dateTime': start_dt.isoformat(),
            'timeZone': timezone,
        },
        'end': {
            'dateTime': end_dt.isoformat(),
            'timeZone': timezone,
        },
    }
    
    if meeting_data.get('meeting_link'):
        event['conferenceData'] = {
            'entryPoints': [{
                'entryPointType': 'video',
                'uri': meeting_data['meeting_link'],
            }]
        }
    
    return service.events().insert(
        calendarId='primary',
        body=event,
        conferenceDataVersion=1 if meeting_data.get('meeting_link') else 0
    ).execute()

def disconnect_user(user_email):
    """Properly clear tokens from database on disconnect"""
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == user_email).first()
        if user:
            user.access_token = None
            user.refresh_token = None
            db.commit()

# ============ MAIN APP ============
st.set_page_config(page_title="Email Calendar Sync", page_icon="📧", layout="wide")
st.title("📧 AI Email-to-Calendar Sync")

# --- OAuth Flow ---
if 'credentials' not in st.session_state:
    # FIX #1: Properly reconstruct the redirect URL from query params
    query_string = urlencode(st.query_params.to_dict())
    current_url = f"{REDIRECT_URI}?{query_string}" if query_string else REDIRECT_URI
    
    if 'code' in st.query_params:
        flow = get_oauth_flow()
        flow.fetch_token(authorization_response=current_url)
        
        credentials = flow.credentials
        st.session_state.credentials = credentials
        
        # Get user email
        oauth_service = build('oauth2', 'v2', credentials=credentials)
        user_info = oauth_service.userinfo().get().execute()
        user_email = user_info['email']
        st.session_state.user_email = user_email
        
        # Save to DB
        with SessionLocal() as db:
            user = db.query(User).filter(User.email == user_email).first()
            if not user:
                db.add(User(
                    email=user_email,
                    access_token=credentials.token,
                    refresh_token=credentials.refresh_token
                ))
            else:
                user.access_token = credentials.token
                user.refresh_token = credentials.refresh_token
            db.commit()
        
        # Clear query params from URL
        st.query_params.clear()
        st.rerun()
    else:
        st.info("Connect your Google Account to start syncing meetings!")
        flow = get_oauth_flow()
        auth_url, _ = flow.authorization_url(
            access_type='offline', 
            include_granted_scopes='true',
            prompt='consent'  # Forces refresh token every time
        )
        st.markdown(
            f'<a href="{auth_url}" target="_self">'
            f'<button style="padding: 12px 24px; font-size: 16px; background-color: #4285f4; '
            f'color: white; border: none; border-radius: 5px; cursor: pointer;">'
            f'Sign in with Google</button></a>',
            unsafe_allow_html=True
        )

else:
    # --- User is logged in ---
    credentials = refresh_credentials_if_needed(st.session_state.credentials)
    user_email = st.session_state.user_email
    
    st.success(f"✅ Logged in as {user_email}")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        st.toggle("Enable Auto-Scan", value=True)
        
        if st.button("🚪 Disconnect"):
            # FIX #6: Actually clear tokens from DB
            disconnect_user(user_email)
            del st.session_state.credentials
            del st.session_state.user_email
            st.rerun()
    
    tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Scan Emails"])
    
    with tab1:
        st.subheader("Recently Created Events")
        with SessionLocal() as db:  # FIX #4: Context manager
            events = db.query(CalendarEvent).filter(
                CalendarEvent.user_email == user_email
            ).order_by(CalendarEvent.created_at.desc()).limit(10).all()
        
        if events:
            for event in events:
                st.write(f"**{event.title}**")
                st.write(f"📅 {event.date} at {event.time}")
                if event.meeting_link:
                    st.write(f"🔗 {event.meeting_link}")
                st.divider()
        else:
            st.info("No events created yet.")
    
    with tab2:
        st.subheader("Scan Your Emails")
        
        if st.button("🔍 Scan Last 10 Unread Emails"):
            with st.spinner("Scanning emails..."):
                gmail_service = build('gmail', 'v1', credentials=credentials)
                calendar_service = build('calendar', 'v3', credentials=credentials)
                
                results = gmail_service.users().messages().list(
                    userId='me', maxResults=10, q='is:unread'
                ).execute()
                
                messages = results.get('messages', [])
                
                if not messages:
                    st.info("No unread emails found!")
                else:
                    # FIX #4: Single DB session for the whole loop
                    with SessionLocal() as db:
                        meetings_found = 0
                        
                        for msg in messages:
                            email_id = msg['id']
                            
                            if db.query(ProcessedEmail).filter(
                                ProcessedEmail.email_id == email_id
                            ).first():
                                continue
                            
                            # FIX #5: Fetch full email body, not just snippet
                            full_message = gmail_service.users().messages().get(
                                userId='me', id=email_id, format='full'
                            ).execute()
                            
                            email_body = get_email_body(full_message)
                            
                            if not email_body.strip():
                                db.add(ProcessedEmail(email_id=email_id, user_email=user_email))
                                continue
                            
                            meeting_data = analyze_email_with_ai(email_body)
                            
                            if meeting_data.get('is_meeting') and meeting_data.get('date') and meeting_data.get('time'):
                                try:
                                    # FIX #2: Proper datetime handling
                                    event = create_calendar_event(calendar_service, meeting_data)
                                    
                                    db.add(CalendarEvent(
                                        id=event['id'],
                                        user_email=user_email,
                                        title=meeting_data.get('title', 'Meeting'),
                                        date=meeting_data['date'],
                                        time=meeting_data['time'],
                                        meeting_link=meeting_data.get('meeting_link', '')
                                    ))
                                    
                                    meetings_found += 1
                                    st.success(
                                        f"✅ **{meeting_data.get('title')}**\n\n"
                                        f"📅 {meeting_data['date']} at {meeting_data['time']}"
                                    )
                                except ValueError as e:
                                    st.warning(f"⚠️ Could not parse meeting time: {e}")
                                except Exception as e:
                                    st.error(f"Failed to create event: {e}")
                            
                            db.add(ProcessedEmail(email_id=email_id, user_email=user_email))
                        
                        db.commit()
                    
                    if meetings_found == 0:
                        st.info("No meetings found in recent emails.")
                    else:
                        st.success(f"✨ Found and created {meetings_found} meeting(s)!")
