import openai
import os
import phi
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq


from dotenv import load_dotenv


from phi.playground import Playground,serve_playground_app


load_dotenv()
phi.api=os.getenv("PHI_API_KEY")
openai.api_key=os.getenv("OPENAI_API_KEY")


web_search_agent = Agent(
    name="web_search_agent",
    role="Search the web for information",
    
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always provide sources"],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="finance_agent",
    role="Fetch stock data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        )
    ],
    instructions=["Use tables for financial data"],
    show_tool_calls=True,
    markdown=True
)
app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)