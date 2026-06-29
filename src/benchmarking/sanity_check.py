import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environmental configurations from root .env file
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def verify_katibin_link():
    print("Initializing pipeline... Checking OpenRouter connectivity.")
    try:
        # Pinging a standard open-weight model as a cost-effective connection test
        response = client.chat.completions.create(
            model="openrouter/free", 
            messages=[
                {"role": "user", "content": "Confirm connection: Reply with 'Katibin System Operational'"}
            ]
        )
        print("\n[SUCCESS] Katibin connection verified successfully!")
        print(f"Gateway Response: {response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"\n[ERROR] Connection failed. Verify your .env token configuration.\nDetails: {e}")

if __name__ == "__main__":
    verify_katibin_link()