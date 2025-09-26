
#Groq API Key setup
import os
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

#convert image into required format
#Base64 is an encoding scheme that converts binary data (like an image, audio, or file) into text (a string of characters).
import base64

# image_path = "acne.jpg"

#function returns a Base64 string version of the image.
def encode_image(image_path):   
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')

#setup multimodal LLM
from groq import Groq
model="llama-3.2-90b-vision-preview"

query="Is there something wrong with my face?"
#model = "meta-llama/llama-4-maverick-17b-128e-instruct"
# model="meta-llama/llama-4-scout-17b-16e-instruct"
#model = "meta-llama/llama-4-scout-17b-16e-instruct"
#model="llama-3.2-90b-vision-preview" #Deprecated

def analyze_image_with_query(query, model, encoded_image):
    client=Groq()  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content
