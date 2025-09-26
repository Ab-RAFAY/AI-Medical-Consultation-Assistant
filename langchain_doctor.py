from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks import StreamingStdOutCallbackHandler
import os


class DoctorChain:
    def __init__(self):
        # ✅ Use correct Groq endpoint & model
        self.chat = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",  # safer, supported model
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        # ✅ Ensure memory matches prompt variables
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            return_messages=True
        )

        # ✅ Prompt with history + current input
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
What's in this image?. Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""),
            ("system", "Conversation history:\n{chat_history}"),
            ("human", "{input}")
        ])

        # ✅ Conversation chain that plugs everything together
        self.chain = ConversationChain(
            llm=self.chat,
            memory=self.memory,
            prompt=self.prompt,
            verbose=True
        )

    def get_response(self, query, image_data=None):
        if image_data:
            full_query = f"""Patient's query: {query}
            Image analysis required: [Image data in base64: {image_data}]"""
        else:
            full_query = query

        response = self.chain.predict(input=full_query)
        return response.strip()
    
    def save_to_memory(self, user_input: str, ai_output: str):
        """Manually save conversation turns into memory"""
        self.memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )


# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
# from langchain.callbacks import StreamingStdOutCallbackHandler
# import os


# class DoctorChain:
#     def __init__(self):
#         # ✅ Use correct Groq endpoint & model
#         self.chat = ChatGroq(
#             api_key=os.environ.get("GROQ_API_KEY"),
#             model="llama3-70b-8192",  # stable model
#             streaming=True,
#             callbacks=[StreamingStdOutCallbackHandler()]
#         )

#         # ✅ Ensure memory matches prompt variables
#         self.memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             input_key="input",
#             return_messages=True
#         )

#         # ✅ Prompt with history + current input
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
# What's in this image?. Do you find anything wrong with it medically? 
# If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
# your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
# Donot say 'In the image I see' but say 'With what I see, I think you have ....'
# Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
# Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""),
#             ("system", "Conversation history:\n{chat_history}"),
#             ("human", "{input}")
#         ])

#         # ✅ Conversation chain
#         self.chain = ConversationChain(
#             llm=self.chat,
#             memory=self.memory,
#             prompt=self.prompt,
#             verbose=True
#         )

#     def get_response(self, query, image_data=None):
#         """Get AI doctor's response and auto-save conversation"""
#         if image_data:
#             full_query = f"""Patient's query: {query}
#             Image analysis required: [Image data in base64: {image_data}]"""
#         else:
#             full_query = query

#         response = self.chain.predict(input=full_query).strip()

#         # ✅ Automatically save user + AI response
#         self.save_to_memory(full_query, response)

#         return response
    
#     def save_to_memory(self, user_input: str, ai_output: str):
#         """Manually save conversation turns into memory (also used internally)"""
#         self.memory.save_context(
#             {"input": user_input},
#             {"output": ai_output}
#         )
