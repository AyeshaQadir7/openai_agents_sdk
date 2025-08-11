import chainlit as cl
from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
from agents.tool import function_tool
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Step 1: Initialize the external client/provider with Gemini API key
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 2: Create model using the external client
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

# Config: Define the run configuration at Run level
run_config = RunConfig(
    model=model, model_provider=external_client, tracing_disabled=True
)


# Function Tool
@function_tool("get_weather")
def get_weather(location: str) -> str:
    """
    Fetch the weather for a given location.
    """
    return f"The weather in {location} is sunny with  22 degrees C."


# Step 3: Create the agent with model and instructions
agent = Agent(
    name=" Agent",
    instructions="You are a helpful agent. use get_weather tool to get temparture of the location.",
    tools=[get_weather],
)


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! How can I help you?").send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})

    result = Runner.run_streamed(
        starting_agent=agent,
        input=history,
        run_config=run_config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    # await cl.Message(content=result.final_output).send()
