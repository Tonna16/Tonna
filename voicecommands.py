import streamlit as st
import streamlit.components.v1 as components

def voice_commands_page():
    st.title("Voice Commands")
    st.write("Use your voice to control the app or ask for energy-saving tips.")

    # Embedded HTML/JavaScript for voice recognition
    voice_html = """
    <html>
      <head>
        <script>
          window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (window.SpeechRecognition) {
              var recognition = new SpeechRecognition();
              recognition.lang = 'en-US';
              recognition.interimResults = false;
              recognition.maxAlternatives = 1;
              function startListening() {
                recognition.start();
              }
              recognition.onresult = function(event) {
                var transcript = event.results[0][0].transcript;
                document.getElementById("voiceResult").value = transcript;
              };
              recognition.onerror = function(event) {
                document.getElementById("voiceResult").value = "Error: " + event.error;
              };
          } else {
              document.getElementById("voiceResult").value = "Your browser does not support Speech Recognition.";
          }
        </script>
      </head>
      <body>
        <button onclick="startListening()">Start Listening</button><br/><br/>
        <textarea id="voiceResult" rows="4" cols="50" placeholder="Your voice command will appear here..."></textarea>
      </body>
    </html>
    """
    components.html(voice_html, height=250)

    # User input field
    voice_command = st.text_input("Enter or paste your voice command here:")
    
    if st.button("Process Command"):
        if voice_command:
            # In production, connect to an AI API (e.g., OpenAI) for real processing.
            answer = f"Simulated response: '{voice_command}' command received."
            st.write("Response:", answer)
        else:
            st.error("Please provide a voice command.")
