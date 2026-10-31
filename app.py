import streamlit as st
import os
import json
import base64
from datetime import datetime, timedelta
from urllib.parse import urlencode
from dotenv import load_dotenv

# Google libraries
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Multi-AI support
import litellm

# Database
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Text
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
    ai_provider = Column(String, default="gemini/gemini-1.5-flash")
    ai_api_key = Column(String)
    ai_base_url = Column(String)
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
DEFAULT_AI_PROVIDER = os.getenv("DEFAULT_AI_PROVIDER", "gemini/gemini-1.5-flash")
DEFAULT_AI_API_KEY = os.getenv("DEFAULT_AI_API_KEY")

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

# ============ AI PROVIDER OPTIONS ============
AI_PROVIDERS = {
    "Google Gemini (Free)": {
        "model": "gemini/gemini-1.5-flash",
        "needs_api_key": True,
        "get_key_url": "https://aistudio.google.com/app/apikey",
        "description": "Free tier, fast, good for meetings"
    },
    "OpenAI GPT-4o": {
        "model": "gpt-4o",
        "needs_api_key": True,
        "get_key_url": "https://platform.openai.com/api-keys",
        "description": "Best accuracy, paid"
    },
    "Anthropic Claude 3.5 Sonnet": {
        "model": "claude-3-5-sonnet-20241022",
        "needs_api_key": True,
        "get_key_url": "https://console.anthropic.com/",
        "description": "Excellent at understanding context"
    },
    "Groq Llama 3.3 (Free)": {
        "model": "groq/llama-3.3-70b-versatile",
        "needs_api_key": True,
        "get_key_url": "https://console.groq.com/keys",
        "description": "Very fast, free tier"
    },
    "Ollama (Local)": {
        "model": "ollama/llama3.2",
        "needs_api_key": False,
        "get_key_url": "https://ollama.ai/download",
        "description": "Run locally, no API key needed",
        "needs_base_url": True,
        "default_base_url": "http://localhost:11434"
    },
    "Custom OpenAI-Compatible": {
        "model": "custom",
        "needs_api_key": True,
        "get_key_url": "",
        "description": "Any OpenAI-compatible API",
        "needs_base_url": True
    }
}

# ============ HELPER FUNCTIONS ============

def get_oauth_flow():
    return Flow.from_client_config(
        CLIENT_CONFIG, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )

def refresh_credentials_if_needed(credentials):
    """Auto-refresh expired access tokens"""
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        # FIX #1: Use proper session context manager
        with SessionLocal() as db:
            user = db.query(User).filter(User.email == st.session_state.user_email).first()
            if user:
                user.access_token = credentials.token
                db.commit()
        st.session_state.credentials = credentials
    return credentials

def get_email_body(message):
    """Decode full email body from Gmail's base64 format"""
    payload = message.get('payload', {})
    
    if payload.get('mimeType') == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        if data:
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    
    parts = payload.get('parts', [])
    for part in parts:
        mime = part.get('mimeType', '')
        if mime == 'text/plain':
            data = part.get('body', {}).get('data', '')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        if mime.startswith('multipart/'):
            for subpart in part.get('parts', []):
                if subpart.get('mimeType') == 'text/plain':
                    data = subpart.get('body', {}).get('data', '')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    
    return message.get('snippet', '')

def analyze_email_with_ai(email_content, provider_config, api_key=None, base_url=None):
    """Use LiteLLM to support any AI provider"""
    
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
        call_params = {
            "model": provider_config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts meeting information from emails. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
        }
        
        if api_key:
            call_params["api_key"] = api_key
        
        if base_url:
            call_params["api_base"] = base_url
        
        response = litellm.completion(**call_params)
        content = response.choices[0].message.content
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                return {"is_meeting": False}
                
    except Exception as e:
        st.error(f"AI provider error: {e}")
        return {"is_meeting": False}

def validate_meeting_data(meeting_data):
    """FIX #6: Validate AI response before creating calendar event"""
    if not meeting_data.get('is_meeting'):
        return False, "Not a meeting"
    
    if not meeting_data.get('date'):
        return False, "Missing date"
    
    if not meeting_data.get('time'):
        return False, "Missing time"
    
    # Validate date format
    try:
        datetime.strptime(meeting_data['date'], "%Y-%m-%d")
    except ValueError:
        return False, f"Invalid date format: {meeting_data['date']}"
    
    # Validate time format
    try:
        datetime.strptime(meeting_data['time'], "%H:%M")
    except ValueError:
        return False, f"Invalid time format: {meeting_data['time']}"
    
    return True, "Valid"

def parse_meeting_datetime(date_str, time_str, duration_hours=1):
    """Safely parse date/time and handle day rollover"""
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
    """Clear tokens from database on disconnect"""
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == user_email).first()
        if user:
            user.access_token = None
            user.refresh_token = None
            db.commit()

# ============ MAIN APP ============
st.set_page_config(page_title="Email Calendar Sync", page_icon="📧", layout="wide")
st.title("📧 AI Email-to-Calendar Sync")

# FIX #1: Initialize session state properly
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.credentials = None
    st.session_state.user_email = None
    st.session_state.ai_provider_name = None
    st.session_state.ai_api_key = None
    st.session_state.ai_base_url = None

# --- OAuth Flow ---
if not st.session_state.authenticated:
    # FIX #3: More robust OAuth callback handling
    if 'code' in st.query_params:
        try:
            # Reconstruct the full redirect URL
            query_params = st.query_params.to_dict()
            query_string = urlencode(query_params)
            authorization_response = f"{REDIRECT_URI}?{query_string}"
            
            flow = get_oauth_flow()
            flow.fetch_token(authorization_response=authorization_response)
            
            credentials = flow.credentials
            
            # Get user email
            oauth_service = build('oauth2', 'v2', credentials=credentials)
            user_info = oauth_service.userinfo().get().execute()
            user_email = user_info['email']
            
            # FIX #1: Set all session state at once
            st.session_state.authenticated = True
            st.session_state.credentials = credentials
            st.session_state.user_email = user_email
            
            # Save to DB
            with SessionLocal() as db:
                user = db.query(User).filter(User.email == user_email).first()
                if not user:
                    db.add(User(
                        email=user_email,
                        access_token=credentials.token,
                        refresh_token=credentials.refresh_token,
                        ai_provider=DEFAULT_AI_PROVIDER,
                        ai_api_key=DEFAULT_AI_API_KEY
                    ))
                    # Load default AI settings into session
                    st.session_state.ai_provider_name = "Google Gemini (Free)"
                    st.session_state.ai_api_key = DEFAULT_AI_API_KEY
                    st.session_state.ai_base_url = None
                else:
                    user.access_token = credentials.token
                    user.refresh_token = credentials.refresh_token
                    # Load user's AI settings into session
                    provider_name = next(
                        (name for name, config in AI_PROVIDERS.items() if config["model"] == user.ai_provider),
                        "Google Gemini (Free)"
                    )
                    st.session_state.ai_provider_name = provider_name
                    st.session_state.ai_api_key = user.ai_api_key
                    st.session_state.ai_base_url = user.ai_base_url
                db.commit()
            
            # Clear query params
            st.query_params.clear()
            st.rerun()
            
        except Exception as e:
            st.error(f"OAuth error: {e}")
            st.query_params.clear()
    
    else:
        st.info("Connect your Google Account to start syncing meetings!")
        flow = get_oauth_flow()
        auth_url, _ = flow.authorization_url(
            access_type='offline', 
            include_granted_scopes='true',
            prompt='consent'
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
        
        # FIX #4 & #5: Store AI settings in session state
        st.subheader("🤖 AI Provider")
        
        provider_names = list(AI_PROVIDERS.keys())
        
        # Find current provider index
        current_provider_name = st.session_state.ai_provider_name or "Google Gemini (Free)"
        current_index = provider_names.index(current_provider_name) if current_provider_name in provider_names else 0
        
        selected_provider_name = st.selectbox(
            "Choose AI Provider",
            provider_names,
            index=current_index,
            key="provider_select"
        )
        
        selected_provider = AI_PROVIDERS[selected_provider_name]
        st.caption(selected_provider["description"])
        
        # API Key input
        if selected_provider["needs_api_key"]:
            if selected_provider["get_key_url"]:
                st.markdown(f"[Get API Key]({selected_provider['get_key_url']})")
            
            api_key_input = st.text_input(
                "API Key",
                value=st.session_state.ai_api_key or "",
                type="password",
                key="api_key_input"
            )
        else:
            api_key_input = None
        
        # Base URL for local/custom models
        base_url_input = None
        if selected_provider.get("needs_base_url"):
            default_url = selected_provider.get("default_base_url", "http://localhost:8000")
            base_url_input = st.text_input(
                "API Base URL",
                value=st.session_state.ai_base_url or default_url,
                key="base_url_input"
            )
        
        # Save settings button
        if st.button("💾 Save AI Settings"):
            # FIX #4 & #5: Update session state
            st.session_state.ai_provider_name = selected_provider_name
            st.session_state.ai_api_key = api_key_input
            st.session_state.ai_base_url = base_url_input
            
            # Save to DB
            with SessionLocal() as db:
                user = db.query(User).filter(User.email == user_email).first()
                if user:
                    user.ai_provider = selected_provider["model"]
                    user.ai_api_key = api_key_input
                    user.ai_base_url = base_url_input
                    db.commit()
            st.success("AI settings saved!")
        
        st.divider()
        
        st.toggle("Enable Auto-Scan", value=True)
        
        if st.button("🚪 Disconnect"):
            disconnect_user(user_email)
            # Clear all session state
            st.session_state.authenticated = False
            st.session_state.credentials = None
            st.session_state.user_email = None
            st.session_state.ai_provider_name = None
            st.session_state.ai_api_key = None
            st.session_state.ai_base_url = None
            st.rerun()
    
    tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Scan Emails"])
    
    with tab1:
        st.subheader("Recently Created Events")
        with SessionLocal() as db:
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
        
        # FIX #4 & #5: Use session state for AI settings
        selected_provider_name = st.session_state.ai_provider_name or "Google Gemini (Free)"
        selected_provider = AI_PROVIDERS[selected_provider_name]
        api_key_to_use = st.session_state.ai_api_key
        base_url_to_use = st.session_state.ai_base_url
        
        st.info(f"Using: **{selected_provider_name}** ({selected_provider['model']})")
        
        if st.button("🔍 Scan Last 10 Unread Emails"):
            # Validate AI settings
            if selected_provider["needs_api_key"] and not api_key_to_use:
                st.error("❌ Please enter your API key in the sidebar and click 'Save AI Settings'!")
            else:
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
                        with SessionLocal() as db:
                            meetings_found = 0
                            errors = []
                            
                            for msg in messages:
                                email_id = msg['id']
                                
                                if db.query(ProcessedEmail).filter(
                                    ProcessedEmail.email_id == email_id
                                ).first():
                                    continue
                                
                                full_message = gmail_service.users().messages().get(
                                    userId='me', id=email_id, format='full'
                                ).execute()
                                
                                email_body = get_email_body(full_message)
                                
                                if not email_body.strip():
                                    db.add(ProcessedEmail(email_id=email_id, user_email=user_email))
                                    continue
                                
                                meeting_data = analyze_email_with_ai(
                                    email_body,
                                    selected_provider,
                                    api_key=api_key_to_use,
                                    base_url=base_url_to_use
                                )
                                
                                # FIX #6: Validate meeting data before creating event
                                is_valid, error_msg = validate_meeting_data(meeting_data)
                                
                                if is_valid:
                                    try:
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
                                    except Exception as e:
                                        errors.append(f"Failed to create event for '{meeting_data.get('title')}': {e}")
                                else:
                                    if meeting_data.get('is_meeting'):
                                        errors.append(f"Skipped meeting due to: {error_msg}")
                                
                                db.add(ProcessedEmail(email_id=email_id, user_email=user_email))
                            
                            db.commit()
                        
                        if meetings_found == 0 and not errors:
                            st.info("No meetings found in recent emails.")
                        elif meetings_found > 0:
                            st.success(f"✨ Found and created {meetings_found} meeting(s)!")
                        
                        if errors:
                            st.warning(f"⚠️ {len(errors)} error(s) occurred:")
                            for error in errors[:5]:  # Show first 5 errors
                                st.write(f"- {error}")
