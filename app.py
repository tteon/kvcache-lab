import chainlit as cl
from chat_service import AdapterFactory

# Default configurations
DEFAULT_URL = "http://localhost:8000/chat"

@cl.on_chat_start
async def start():
    # Initialize settings
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="mode",
                label="Connection Mode",
                values=["Generic API", "Online Serving Platform"],
                initial_index=0,
            ),
            cl.input_widget.TextInput(
                id="target_url",
                label="Target Server URL",
                initial=DEFAULT_URL,
            ),
            cl.input_widget.TextInput(
                id="api_key",
                label="API Key (Optional)",
                initial="",
                type="password"
            ),
        ]
    ).send()

    # Store settings in user session for access in on_message
    cl.user_session.set("settings", settings)
    
    await cl.Message(
        content=f"Welcome! \n\nI am configured to chat with: `{DEFAULT_URL}`.\n\nYou can change the **Target Server URL** and **Connection Mode** in the Chat Settings (bottom left)."
    ).send()

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    await cl.Message(
        content=f"Settings updated!\nTarget: `{settings['target_url']}`\nMode: `{settings['mode']}`"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")
    
    # Fallback if settings haven't been loaded yet (shouldn't happen with on_chat_start)
    if not settings:
        settings = {
            "mode": "Generic API",
            "target_url": DEFAULT_URL,
            "api_key": ""
        }

    mode = settings.get("mode", "Generic API")
    url = settings.get("target_url", DEFAULT_URL)
    api_key = settings.get("api_key", "")
    
    # Instantiate adapter (re-creating per message allows dynamic settings changes)
    adapter = AdapterFactory.get_adapter(mode, url, api_key)
    
    msg = cl.Message(content="")
    await msg.send()
    
    # Run sync network call in thread to not block async loop
    response = await cl.make_async(adapter.send_message)(message.content)
    
    msg.content = response
    await msg.update()
