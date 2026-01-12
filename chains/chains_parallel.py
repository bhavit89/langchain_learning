from dotenv import load_dotenv 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda ,RunnableParallel
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


summary_template = ChatPromptTemplate(
    [("system", "you are a movie critic"),
     ("human", "Provide a brief summary of the movie {movie_name}")
    ]
)

# defining plot analysis step 

def plot_analyze(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system","you are movie critic"),
            ("human", "Analyze about the plot :{plot} what are the strengths and weakness")
        ]

    )

    return plot_template.format_prompt(plot=plot)


# defining character analysis plot 

def character_analysis(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a movie critic"),
            ("human", "Analyze the {characters} and tell their strenght and weakness")
        ]

    )

    return character_template.format_prompt(characters=characters)


# combine verdicts 

def combine_verdicts(plot_analysis , character_analysis):
    return f"plot analysis:\n{plot_analysis}\n\n Character Analysis:\n{character_analysis}"


plot_branch_chain = (
    RunnableLambda(lambda x : plot_analyze(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: character_analysis(x)) | model |StrOutputParser()
)

chain = (
    summary_template
    | model 
    | StrOutputParser()
    | RunnableParallel(branches={"plot":plot_branch_chain,"characters":character_branch_chain})
    | RunnableLambda(lambda x : combine_verdicts(x["branches"]["plot"] ,x["branches"]["characters"]))
)


result = chain.invoke({"movie_name":"Fury"})
print(result)