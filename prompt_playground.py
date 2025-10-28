
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
HF_TOKEN = "your token"

DEFULAT_SYSTEM_MESSAGE = "you are an helpful AI assistant"
DEFULAT_TEMPERATURE = 0.7
log_file = "prompt_playground.txt"
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    max_new_tokens=100,
    temperature=DEFULAT_TEMPERATURE,
    huggingfacehub_api_token=HF_TOKEN,
)
model = ChatHuggingFace(llm=llm)


def save_log(sys_msg, temp, prompt, AI_replay):
    with open(log_file, "a", encoding='utf-8')as f:
        log_entry = (
            f"{datetime.now()}\n"
            f"system message :{sys_msg}\n"
            f"Temperature : {temp}\n"
            f"user prompt :{prompt}\n"
            f"model replay :{AI_replay}\n"
            f"{'-'*60}\n"
        )
        f.write(log_entry)


def prompt_playground():
    print("prompt_playground")
    sys_mes = DEFULAT_SYSTEM_MESSAGE
    while True:
        print(f"system prompt :{sys_mes}")
        print("type system to change , type temp to change temparature, and quit to exit")
        user_input = input("You:").strip()
        if user_input.lower() == "exit":
            print("Goodbye")
            break
        if user_input.lower() == "system":
            sys_mes = input("New").strip() or sys_mes
            continue
        if user_input.lower() == "temp":
            try:
                temp = float(input("set temparature").strip())
                model.llm.temperature = temp
                print(f"temparature set to {temp}")
            except Exception as e:
                print(f"Invalid temperature value:{e}")
            continue
        messages = [SystemMessage(content=sys_mes),
                    HumanMessage(content=user_input)]
        result = model.invoke(messages)
        reply = result.content
        print(f"\n model reply:\n {reply}")
        save_log(user_input, sys_mes, reply, model.llm.temperature)


if __name__ == "__main__":
    prompt_playground()
