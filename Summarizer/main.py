import requests
import json
import os

# Ollama local API endpoint
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3"  # Change if you're using another model

# System prompt to instruct the model
SYSTEM_PROMPT = """
You are a medical assistant. Your job is to extract and summarize patient information from the given text.

Return only valid JSON with the following format:
{
  "name": "",
  "age": "",
  "gender": "",
  "diseases": [],
  "symptoms": [],
  "duration": "",
  "other_notes": ""
}

- Go through the provided data and extract all the fields in the data.
- Return each and every field present in the Data , Do not miss any field. 
- Extract `duration` if the user mentions things like "for 2 weeks", "since last year", etc.
- In `other_notes`, include medications, lifestyle habits, family history, recent diagnoses, or appointments.
- If any field is missing in the input, leave it empty. Do not return explanations. Return only the JSON.
"""

# Function to send patient input to Ollama
def get_patient_summary(patient_text: str):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": patient_text}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print("❌ Error:", e)
        print("⚠️ Raw response content:", result.get("message", {}).get("content", "N/A"))
        return None

# Main function
def main():
    file_path = "patient_info.txt"

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        patient_text = f.read().strip()

    if not patient_text:
        print("⚠️ File is empty.")
        return

    print("📤 Sending patient info to Ollama for summarization...\n")
    summary = get_patient_summary(patient_text)

    if summary:
        print("✅ Summary Extracted:\n")
        print(json.dumps(summary, indent=2))
    else:
        print("❌ Failed to summarize patient info.")

# Entry point
if __name__ == "__main__":
    main()
