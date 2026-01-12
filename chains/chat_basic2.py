from  dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os 

load_dotenv()
print("[----- LIB IMPORTED --------]")


api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    api_key=api_key,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system" , "you are a expert who knows facts about {cars}"),
        ("human" , "Tell me {facts_counts} facts.")
    ]

)

# create the combined chain using Langhcain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
result = chain.invoke({"cars":"mercedes","facts_counts":2})
print(result)