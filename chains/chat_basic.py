from  dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os 

load_dotenv()
print("[----- LIB IMPORTED --------]")


llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to spanish. Translate the user sentence.",
    ),
    ("human", "Tell me a joke."),
]
ai_msg = llm.invoke(messages)
ai_msg

print(ai_msg.content)