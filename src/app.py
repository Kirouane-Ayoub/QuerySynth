import mesop as me
import mesop.labs as mel
from agent import agent


@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/chat",
    title="QuerySynth Demo Chat",
)
def page():
    mel.chat(transform, title="QuerySynth", bot_user="QuerySynth bot")


def transform(input: str, history: list[mel.ChatMessage]):
    prompt = input  # Use the input as the prompt for the agent
    response = agent.chat(prompt)  # Get the response from the agent
    yield response.response
