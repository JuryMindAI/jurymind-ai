from jurymind.llm.openai import OpenAILLM
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from jurymind-ai!")
    client = OpenAILLM(api_key="")
    
    response = client.completion("What is 2 + 2?")
    
    print(response)
    
    


if __name__ == "__main__":
    main()
