from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence
load_dotenv()

prompt1=PromptTemplate(
    template="Generate a tweet about {topic}.",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Generate a LinkedIn post about {topic}.",
    input_variables=["topic"]
)

parser=StrOutputParser()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

chain=RunnableParallel({
    'Tweet': RunnableSequence(prompt1, model, parser),
    'LinkedIn Post': RunnableSequence(prompt2, model, parser)
})
print(chain.invoke({"topic":"LangChain"}))

chain.get_graph().print_ascii()
