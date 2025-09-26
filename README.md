# AI-Medical-Consultation-Assistant
An AI-powered multimodal healthcare assistant designed to make medical consultations more accessible—especially for elderly patients. The system supports text, voice, and medical image and reports inputs, processes them through advanced AI models, and provides real-time text and voice-based medical responses.
This project provides a voice and image-enabled AI doctor assistant with a simple, chat-like interface. Patients can speak their symptoms, upload medical images, or type messages. The AI processes inputs, maintains conversational memory, and delivers accurate, accessible responses—both in text and speech.**

# Tech Stack:
Frontend/UI:
  Gradio (dark-theme, chat interface)

Voice Processing:
  Whisper (speech-to-text)
  
gTTS (text-to-speech):
  FFmpeg & PortAudio

Image Processing:
  Base64 encoding

Groq LLM for analysis

AI Core:
  LangChain + Groq LLM for reasoning & medical consultation

APIs:
  Groq API (LLM services)
  Google TTS
  Whisper

Memory Management:
  Conversational buffer memory for context-aware interactions

# Pipeline
Input Layer:
  Text, Voice, or Image provided by the patient.
Processing Layer:
  Voice → text, image → encoded data, text → processed query.
AI Processing:
  LangChain + Groq LLM analyze and generate medical responses.
Output Layer:
  Response displayed in chat + optional audio playback via TTS.

# Project Structure
├── gradio_app.py             # User Interface (chat, voice, image input)
├── voice_of_the_patient.py   # Voice input & transcription
├── voice_of_the_doctor.py    # Text-to-speech output
├── brain_of_the_doctor.py    # Image processing & encoding
├── langchain_doctor.py       # AI doctor chain using LangChain + Groq LLM

