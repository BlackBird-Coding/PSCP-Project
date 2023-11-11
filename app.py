import gradio as gr
import os
from ctransformers import AutoModelForCausalLM
from googletrans import Translator


from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.retrievers import WikipediaRetriever
from langchain.vectorstores import Chroma

loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(data)

embedding = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")

persist_directory = "db"
if os.path.exists(persist_directory):
    print("Loading from disk")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
else:
    print("Creating new database")
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embedding, persist_directory=persist_directory
    )
    vectordb.persist()

retriever = vectordb.as_retriever()

translator = Translator()

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/dolphin-2.2.1-mistral-7B-GGUF",
    model_file="dolphin-2.2.1-mistral-7b.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=20,
    context_length=4096 * 2,
    stop=["<|im_start|>", "<|im_end|>"],
)


# template_with_context = """<s>[INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, If you don't know the answer to a question, please don't share false information. Answer exactly in few words from the context
# Context for question below:
# {context}
# Question answer by upper context:
# {question} [/INST] </s>
# """

# template_with_history = """<s>[INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, If you don't know the answer to a question, please don't share false information.
# {question} [/INST] </s>
# """

template_with_context = """<|im_start|>system
You are Problem Solving and Computer Programming Class Chatbot. If you don't know the answer to a question, please don't share false information. Answer exactly in few words from the context
<|im_start|>user
history: {history}
context: {context}
{question} <|im_end|>
<|im_start|>assistant
"""

template_without_context = """<|im_start|>system
You are Problem Solving and Computer Programming Class Chatbot. If you don't know the answer to a question, please don't share false information.
{history}
<|im_start|>user
{question} <|im_end|>
<|im_start|>assistant
"""


class ChitChat:
    def __init__(self, translator, llm, template):
        self.translator = translator
        self.llm = llm
        self.template = template

    def preprocess(self, history):
        new_history = ""
        for message in history[-5:]:
            q = message[0]
            a = message[1]
            new_history += f"<|im_start|>user\n{self.translator.translate(q).text}\n<|im_start|>assistant\n{a.split('คำแปล: ')[0]}"
        return new_history

    def predict(self, message, history):
        message = self.translator.translate(message, dest="en").text
        history = self.preprocess(history)
        bot_message = self.llm(
            self.template.format_map(
                {
                    "question": message,
                    "history": history,
                }
            ),
            stream=True,
        )

        partial_message = ""
        for character in bot_message:
            partial_message += character
            yield partial_message
        yield partial_message + "\nคำแปล: " + self.translator.translate(
            partial_message, dest="th"
        ).text

    def gradio_block(self):
        return gr.ChatInterface(self.predict)


class ChatWithClass:
    def __init__(self, translator, llm, template, retriever):
        self.translator = translator
        self.llm = llm
        self.template = template
        self.retriever = retriever

    def predict(self, message, history):
        message = self.translator.translate(message, dest="en").text

        bot_message = self.llm(
            self.template.format_map(
                {
                    "question": message,
                    "context": " ".join(
                        [
                            x.page_content
                            for x in self.retriever.get_relevant_documents(
                                message,
                            )
                        ]
                    ),
                    "history": " ".join(history[-5:]),
                }
            ),
            stream=True,
        )

        partial_message = ""
        for character in bot_message:
            partial_message += character
            yield partial_message

    def gradio_block(self):
        return gr.ChatInterface(self.predict)


class ChatWithWiki:
    def __init__(self, translator, llm, template):
        self.translator = translator
        self.llm = llm
        self.template = template
        self.retriever = WikipediaRetriever()

    def predict(self, message, history):
        message = self.translator.translate(message, dest="en").text

        bot_message = self.llm(
            self.template.format_map(
                {
                    "question": message,
                    "context": " ".join(
                        [
                            x.page_content
                            for x in self.retriever.get_relevant_documents(
                                message,
                            )
                        ]
                    ),
                    "history": " ".join(history[-5:]),
                }
            ),
            stream=True,
        )

        partial_message = ""
        for character in bot_message:
            partial_message += character
            yield partial_message

    def gradio_block(self):
        return gr.ChatInterface(self.predict)


chit_chat = ChitChat(translator, llm, template_without_context)
chat_with_class = ChatWithClass(translator, llm, template_with_context, retriever)
chat_with_wiki = ChatWithWiki(translator, llm, template_with_context)


app = gr.TabbedInterface(
    [
        chit_chat.gradio_block(),
        chat_with_class.gradio_block(),
        chat_with_wiki.gradio_block(),
    ],
    ["Chit Chat", "PSCP Chat", "Wikipedia Chat"],
    title="PSCP Chatbot",
)

if __name__ == "__main__":
    app.launch()
