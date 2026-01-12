from langchain_core.messages import SystemMessage ,HumanMessage , AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama-3.1-8b-instant",
                 temperature=0,
                 max_tokens=None,
                 api_key=api_key,
                 timeout=None,
                 max_retries=2,
                 )


chat_history = []
system_message = SystemMessage(content="you are help  AI  assistant")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print( f"AI : {response}")

print("[---- CHAT ENDED -----]")