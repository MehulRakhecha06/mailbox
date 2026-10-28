# Autonomous Email-to-Calendar AI Agent

An autonomous background worker that watches your Gmail inbox, filters meeting-related communications using Google Gemini, and automatically logs events straight into your Google Calendar with custom reminders.

---

## 🛠️ Step-by-Step Setup Guide

### 1. Google Cloud Console Setup
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project.
3. Enable the **Gmail API** and **Google Calendar API**.
4. Configure the **OAuth consent screen** (set user type to External, add your email as a test user).
5. Create **OAuth Client ID credentials** (Application type: *Desktop App*).
6. Download the generated JSON credentials file, rename it to `credentials.json`, and place it in the root directory of this project.

### 2. Environment Variables Configuration
Copy the template `.env.example` file to `.env`:
```bash
cp .env.example .env
