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

system_message = SystemMessage("you are a expert physicist")
huma_message = HumanMessage("Tell me about time dilation")

messages = [system_message , huma_message]
result = model.invoke(messages)
print("[---Response----]","\n\n")
print(result.content)