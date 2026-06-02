import requests
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.agents import create_agent
from langchain.tools import tool

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
    model = model,
    tools = [get_weather],
    system_prompt = 'You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful.'
)

response = agent.invoke({
    'messages': [
        {'role':'user', 'content': 'What is the weather like in Vienna?'}
    ]
})

print(response['messages'][-1].content)