from  dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda ,RunnableSequence ,RunnableBranch
from langchain_groq import ChatGroq
import os 


load_dotenv()

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

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are and helpful assistant"),
        ("human", "Generate  a thank you note for this positive feedback : {feedback} ."),
    ]
)


negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are and helpful assistant"),
        ("human", "Generate  a response note for this addressing negative feedback : {feedback} ."),
    ]
)

nuetral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are and helpful assistant"),
        ("human", "Generate  a response note for this addressing nuetral feedback : {feedback} ."),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are and helpful assistant"),
        ("human", "Generate  a response note for this addressing escalate feedback to  human agent : {feedback} ."),
    ]
)


classification_template = ChatPromptTemplate.from_messages(
    [
        ("system" ,"you are a helpful AI assistant"),
        ("human","Classify the sentiment of the feedback as positive  , negative, nuetral ,or escalate {feedback}")
    ]
)


# Define conditional branches 

branches = RunnableBranch(
    (
        lambda x: "positve" in x , positive_feedback_template | model | StrOutputParser()
    ),
    
    (
        lambda x: "negative" in x , negative_feedback_template | model | StrOutputParser()
    ),

    (
        lambda x: "nuetral" in x , nuetral_feedback_template | model | StrOutputParser()
    ),

    escalate_feedback_template | model | StrOutputParser(),

    )


classification_chain = classification_template | model | StrOutputParser()
chain = classification_chain | branches 

review = "The Product is teriible it was damaged when the delivered also the quality is  pretty  bad "
result = chain.invoke({"feedback":review})
print(result)