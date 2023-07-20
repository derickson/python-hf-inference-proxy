from lib_webLLM import WebLLM
from langchain import PromptTemplate, LLMChain

llm = WebLLM("http://localhost:8000/falcon")

template_text = """You are an AI assistant that answers questions.\Human: {question}\nAI: """
prompt_template = PromptTemplate(template=template_text, input_variables=["question"])
chain = LLMChain(prompt=prompt_template, llm=llm)

response_text = chain.run(question="What is the capital of the United Kingdom?")

print(response_text)