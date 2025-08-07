# agent_runner.py
import autogen
import re
from ml_model import NoShowPredictor
# Instantiate your ML model
predictor = NoShowPredictor()
llm_config = {
    "cache_seed": 42,
    "config_list": [{
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # dummy key for Ollama
    }]
}
# Function to parse user input and return prediction
def predict_no_show_logic(message: str) -> str:
    try:
        nums = list(map(int, re.findall(r'\d+', message)))
        if len(nums) != 3:
            return "❌ Please provide exactly 3 numbers: age, sms_reminder (1/0), days_between."
        age, sms, days_between = nums
        if sms not in (0, 1):
            return "❌ SMS reminder must be 1 (yes) or 0 (no)."
        result = predictor.predict(age, sms, days_between)
        return f"✅ Prediction: {result}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# User interface agent
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=3,
    code_execution_config=False,
)

# Assistant agent that guides and invokes ML logic
predictor_agent = autogen.AssistantAgent(
    name="PredictorAgent",
    llm_config=llm_config,
    system_message=(
        "You are a helpful assistant that predicts whether a patient will no-show an appointment.\n"
        "Ask the user for three values:\n"
        "1. Age\n"
        "2. SMS reminder (1 for yes, 0 for no)\n"
        "3. Number of days between scheduling and appointment.\n"
        "Once collected, pass them as a string like '45 1 5' to `predict_no_show_logic_interface()` "
        "and respond with the prediction result."
    )
)

# Register prediction logic with the assistant
@predictor_agent.register_for_execution()
def predict_no_show_logic_interface(message: str) -> str:
    """Takes a string like '45 1 5' and returns prediction."""
    return predict_no_show_logic(message)

# Optional: CLI test run
if __name__ == "__main__":
    print("✅ Test Prediction Output:")
    print(f"Patient: Age 20, SMS=0, Days=1 → Prediction:",
    predictor.predict(17, 1, 1))

    # To run chat, uncomment below:
    # user_proxy.initiate_chat(predictor_agent, message="Help me predict no-show for appointment.")
