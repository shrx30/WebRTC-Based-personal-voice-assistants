from langchain_openai import ChatOpenAI

from app.config.settings import settings


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)