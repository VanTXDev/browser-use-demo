import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional

# Third-party imports
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use.agent.views import AgentHistoryList
from browser_use import BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import pyperclip

# Local module imports
from browser_use import Agent

load_dotenv()


@dataclass
class ActionResult:
	is_done: bool
	extracted_content: Optional[str]
	error: Optional[str]
	include_in_memory: bool


@dataclass
class AgentHistoryList:
	all_results: List[ActionResult]
	all_model_outputs: List[dict]


def parse_agent_history(history_str: str) -> None:
	console = Console()

	# Split the content into sections based on ActionResult entries
	sections = history_str.split('ActionResult(')

	for i, section in enumerate(sections[1:], 1):  # Skip first empty section
		# Extract relevant information
		content = ''
		if 'extracted_content=' in section:
			content = section.split('extracted_content=')[1].split(',')[0].strip("'")

		if content:
			header = Text(f'Step {i}', style='bold blue')
			panel = Panel(content, title=header, border_style='blue')
			console.print(panel)
			console.print()


async def run_browser_task(
	task: str,
	choosed_model: str = 'gemini-2.0-flash-exp',
	headless: bool = True,
) -> str:
	load_dotenv()
	api_key = os.getenv('GEMINI_API_KEY')
	if not api_key:
		raise ValueError('GEMINI_API_KEY is not set')

	os.environ['GEMINI_API_KEY'] = api_key

	try:
		browser = Browser(
			config=BrowserConfig(
				new_context_config=BrowserContextConfig(
					viewport_expansion=0,
				),
				headless=headless
			),
		)
		agent = Agent(
			task=task,
			llm=ChatGoogleGenerativeAI(model=choosed_model, api_key=api_key),
			browser=browser,
		)
		history: AgentHistoryList = await agent.run(max_steps=5)
		return history.final_result()
	except Exception as e:
		return f'Error: {str(e)}'


def create_ui():
	with gr.Blocks(title='Browser Use GUI') as interface:
		gr.Markdown('# Browser Use Task Automation')

		with gr.Row():
			with gr.Column():
				task = gr.Textbox(
					label='Task Description',
					placeholder='E.g., Find flights from New York to London for next week',
					lines=3,
				)
				model = gr.Dropdown(choices=['gemini-2.0-flash-exp', 'gpt-4', 'gpt-3.5-turbo'], label='Model', value='gemini-2.0-flash-exp',)
				headless = gr.Checkbox(label='Run Headless', value=True)
				submit_btn = gr.Button('Run Task')

			with gr.Column():
				output = gr.Textbox(label='Output', lines=10, interactive=False, show_copy_button=True)

		submit_btn.click(
			fn=lambda *args: asyncio.run(run_browser_task(*args)),
			inputs=[task, model, headless],
			outputs=output,
		)

	return interface


if __name__ == '__main__':
	demo = create_ui()
	demo.launch()