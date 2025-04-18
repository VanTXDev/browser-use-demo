"""
Show how to use custom outputs.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

from browser_use.agent.service import Agent
from browser_use.controller.service import Controller

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
	raise ValueError('GEMINI_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

browser = Browser(
	config=BrowserConfig(
		new_context_config=BrowserContextConfig(
			viewport_expansion=0,
		)
	)
)
controller = Controller()


class Destination (BaseModel):
	name: str
	price: str
	address: str
	services: str
	rating: str


class Destinations(BaseModel):
	destination: List[Destination]


@controller.action(description='Save models', param_model=Destinations)
def save_models(params: Destination) -> None:
	with open('destinations.txt', 'a') as f:
		for destination in params.destination:
			f.write(f'{destination.name}|{destination.price}|{destination.address}|{destination.services}|{destination.rating}\n')


async def main():
	agent = Agent(
		task="""Go to google.com, search and get informations about hotels, homestay arround Ninh Hai district in Ninh Thuan province.
		Then summarize their main informations with name, price, address, services, rating on the first page. 
		Then save informations to file!
		Finally generate a marketing plan for my homestay!""",
		llm=llm,
		controller=controller,
	)

	await agent.run(max_steps=20)
	await browser.close()


if __name__ == '__main__':
	asyncio.run(main())