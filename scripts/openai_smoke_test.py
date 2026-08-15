from openai import OpenAI
client = OpenAI()
response = client.responses.create(
    model="gpt-5.4",
    input = "Write one sentence saying the API connection is working."
)
print(response.output_text)
