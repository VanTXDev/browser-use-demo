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

from browser_use import BrowserConfig
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


class Hotel(BaseModel):
	name: str
	price: int
	room: str
	address: str
	link: str
	strengths: str
	strengths: str


class Hotels(BaseModel):
	hotels: List[Hotel]


@controller.action(description='Save models', param_model=Hotels)
def save_models(params: Hotels) -> None:
	with open('list_hotels.txt', 'a') as f:
		for hotel in params.hotels:
			f.write(f'{hotel.name}|{hotel.price}|{hotel.room}|{hotel.address}|{hotel.link}|{hotel.strengths}|{hotel.weakness}\n')


async def main():
	agent = Agent(
		task="""Go to google.com, search and get informations about top 15 hotels, homestay arround Ninh Hai district in Ninh Thuan province.
		Then summarize their main informations with name, price, room, address, link, strengths and strengths. 
		Then save informations to file!
		""",
		llm=llm,
		controller=controller
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())