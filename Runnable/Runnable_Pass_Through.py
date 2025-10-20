from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence,RunnablePassthrough
load_dotenv()

prompt1=PromptTemplate(
    template="Generate a joke about {topic}.",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Explain the joke about {text}.",
    input_variables=["text"]
)

parser=StrOutputParser()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

chain=RunnableSequence(
    RunnableSequence(prompt1, model, parser),  # joke is generated here
    RunnableParallel({
        'Joke': RunnablePassthrough(),  # joke is passed through here
        'Explanation': RunnableSequence(prompt2, model, parser)
    })
)
print(chain.invoke({"topic":"LangChain"}))

chain.get_graph().print_ascii()
