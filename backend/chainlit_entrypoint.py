# models.py
import logging
import os
import sys
from abc import ABC, abstractmethod

import requests
import torch
import yaml
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai.chat_models import AzureChatOpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
)

import chainlit as cl


@cl.step
def tool():
    return "Response from the tool!"


class BaseModel(ABC):
    @abstractmethod
    def get_response(self, prompt: str):
        raise NotImplementedError("This method should be overridden by subclass")


class ChatBasedGPT4Model(BaseModel):
    @staticmethod
    def get_response(input_dict):
        try:
            with open("./backend/config.yaml", "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")

        llm = AzureChatOpenAI(
            # openai_api_base=config["OPENAI_API_BASE"],
            openai_api_version=config["OPENAI_API_VERSION"],
            deployment_name="gpt-4",
            openai_api_key=config["OPENAI_API_KEY"],
            openai_api_type=config["OPENAI_API_TYPE"],
            azure_endpoint=config["AZURE_ENDPOINT"],
        )

        system_message = SystemMessage(content=input_dict["system_prompt"])
        # AIMessage(content=input_prompt_template.GPT_ASSISTANT_PROMPT_TEMPLATE.format(input=""))
        user_message = HumanMessage(content=input_dict["context_prompt"])

        response = llm(messages=[system_message, user_message])
        if not response.content:
            print("No content in the response")
            return None
        result = response.content
        # print(f"result : {result}")

        return result


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    # Initialize the ChatBasedGPT4Model
    gpt4_model = ChatBasedGPT4Model()

    # 메시지 내용에 따라 처리 방식을 분기
    if "🤖" in message.content:
        input_dict = {
            "system_prompt": "이모티콘을 적극적으로 활용해서 질문에 올바르게 대답하세요.",
            "context_prompt": message.content[1:],  # 이모지 string 제외
        }
    else:  # 일반적인 경우의 처리
        input_dict = {
            "system_prompt": "Please provide a response to the following",  # 요구사항에 따라 시스템 프롬프트 조정
            "context_prompt": message.content,  # 사용자 메시지를 컨텍스트 프롬프트로 사용
        }

    # GPT-4 모델로부터 응답 받기
    gpt4_response = gpt4_model.get_response(input_dict)

    # GPT-4 응답이 있는지 확인
    if gpt4_response:
        # GPT-4 응답을 최종 답변으로 사용자에게 보내기
        await cl.Message(content=gpt4_response).send()
    else:
        # GPT-4로부터 응답을 받지 못한 경우, 대체 메시지 보내기
        await cl.Message(
            content="Sorry, I couldn't get a response. Please try again."
        ).send()
