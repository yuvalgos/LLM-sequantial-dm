import openai
from bit_inversion_envs import VanillaBitInversionEnv
import os


openai.api_key = os.getenv("OPENAI_CRS_KEY")

messages = [
    {"role": "user", "content": "Hello, who are you?"},
]


res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

