from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

from dotenv import load_dotenv

import os

load_dotenv()
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

multi_ai_agent = Agent(
    name="multi_agents",
    role="Combine web search + financial insights",
     model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_search_agent, finance_agent],
    instructions=["Always cite sources","Use tables for financial data"],
    show_tool_calls=True,
    markdown=True
)



multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA",
    stream=True
)
