from  dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
import os 


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="qwen/qwen3-32b",temperature=0,
    max_tokens=None,
    api_key=api_key,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

cars_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system","you like telling facts and  you tells facts about {cars}"),
        ("human", "Tell me {count} facts")
    ]
)


translation_template = ChatPromptTemplate.from_messages(
    [
        ("system","You  are a translator and convert the provide text into {language}"),
        ("human","Translate the following text {language}:{text}")
    ]
)


count_words = RunnableLambda(lambda x : f"word_count:{len(x.split())}\n{x}")
prepare_for_transalation = RunnableLambda(lambda output :{"text":output ,"language":"french"})


chain = cars_facts_template | model | StrOutputParser() |prepare_for_transalation | translation_template | model | StrOutputParser()
result = chain.invoke({"cars":"lamborghini" , "count":2})
print(result)