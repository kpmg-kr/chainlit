import os
from typing import List

import yaml
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import chainlit as cl


@cl.step
def tool():
    return "Response from the tool!"


try:
    with open("./backend/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"Failed to load config: {e}")


# openai_api_base=config["OPENAI_API_BASE"],
os.environ["OPENAI_API_VERSION"] = config["OPENAI_API_VERSION"]
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["OPENAI_API_TYPE"] = config["OPENAI_API_TYPE"]
os.environ["AZURE_OPENAI_ENDPOINT"] = config["AZURE_ENDPOINT"]


llm = AzureChatOpenAI(
    # openai_api_base=config["OPENAI_API_BASE"],
    openai_api_version=config["OPENAI_API_VERSION"],
    deployment_name="gpt-4",
    openai_api_key=config["OPENAI_API_KEY"],
    openai_api_type=config["OPENAI_API_TYPE"],
    azure_endpoint=config["AZURE_ENDPOINT"],
)


class ChatBasedGPT4Model:
    @staticmethod
    def get_response(input_dict):
        system_message = SystemMessage(content=input_dict["system_prompt"])
        user_message = HumanMessage(content=input_dict["context_prompt"])

        response = llm(messages=[system_message, user_message])

        if not response.content:
            print("No content in the response")
            return None

        return response.content


async def process_and_embed_pdf(file_path):
    # PDF 파일을 텍스트로 변환
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    texts = [str(text) for text in texts if text is not None]

    from langchain_openai import AzureOpenAIEmbeddings

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="ada-2",
        openai_api_version=config["OPENAI_API_VERSION"],
    )

    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        AzureChatOpenAI(
            # openai_api_base=config["OPENAI_API_BASE"],
            openai_api_version=config["OPENAI_API_VERSION"],
            deployment_name="gpt-4",
            openai_api_key=config["OPENAI_API_KEY"],
            openai_api_type=config["OPENAI_API_TYPE"],
            azure_endpoint=config["AZURE_ENDPOINT"],
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    cl.user_session.set("chain", chain)


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

    elements = message.elements

    docsearch = None

    file_names = ""

    if elements is not None:
        for element in elements:
            file_name = element.name
            file_path = element.path

            file_names += f"{file_name} is embedded..🤖 \n"

            await process_and_embed_pdf(file_path)

    await cl.Message(content=f"{file_names} ").send()

    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    if chain is not None:
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]  # type: List[Document]
        text_elements = []  # type: List[cl.Text]

        retrival_text = ""

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
                retrival_text += source_doc.page_content

            source_names = [text_el.name for text_el in text_elements]
            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"

            await cl.Message(content=answer, elements=text_elements).send()

            answer += f"\nSources: {', '.join(source_names)}"

            print("answer", answer)

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
            "system_prompt": "당신은 AI 전문가이다. 딥러닝 지식을 최대한 쉽게 친절하게 설명하세요.",  # 요구사항에 따라 시스템 프롬프트 조정
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
