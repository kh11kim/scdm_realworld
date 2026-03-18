# 서버에서 실행: uv add fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
PORT = 8000

class Message(BaseModel):
    text: str

@app.post("/echo")
async def echo(message: Message):
    print(f"받은 메시지: {message.text}")
    # 받은 텍스트를 대문자로 바꿔서 응답 (서버가 일하고 있다는 증거)
    return {"reply": f"SERVER ECHO: {message.text.upper()}"}

if __name__ == "__main__":
    import uvicorn
    # 모든 IP에서 접속 허용
    uvicorn.run(app, host="0.0.0.0", port=PORT)