from  dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda ,RunnableSequence
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

prompt_template = ChatPromptTemplate(
    [
        ("system","you love fact and tell fact about {cars}"),
        ("human","Tell me {count} facts")
    ]
)


format_prompt = RunnableLambda(lambda x : prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x : model.invoke(x.to_messages()))
parse_ouput = RunnableLambda(lambda x : x.content)


chain = RunnableSequence(first=format_prompt , middle=[invoke_model] ,last=parse_ouput)
response = chain.invoke({"cars" : "mercesdes" ,"count": 4})
print(response)