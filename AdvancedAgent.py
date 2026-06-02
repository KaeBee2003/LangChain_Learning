import requests
from dotenv import load_dotenv
load_dotenv()


from dataclasses import dataclass
@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_farenheit: float
    humidity: float


from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature = 0.3
)


from langchain.tools import tool, ToolRuntime
@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


@tool('locate_user', description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'Birmingham'
        case _:
            return 'Unknown'


from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()


from langchain.agents import create_agent
agent = create_agent(
    model = model,
    tools = [get_weather,locate_user],
    system_prompt = 'You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful.',
    context_schema = Context,
    response_format=ResponseFormat,
    checkpointer = checkpointer
)


config = {'configurable': {'thread_id':'weather-session'}}


response = agent.invoke({
    'messages': [
        {'role':'user', 'content': 'What is the weather like?'}
    ]},
    config = config,
    context = Context(user_id='XYZ456')
)


# print(response)
print(response['structured_response'].summary)