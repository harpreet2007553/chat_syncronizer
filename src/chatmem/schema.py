from datetime import datetime
from pydantic import BaseModel


class Message(BaseModel):
    message_id : str
    conversation_id : str
    platform : str
    role : str
    title: str
    text : str
    timestamp : datetime
    
class Conversation(BaseModel):
    conversation_id : str
    platform: str
    title:str
    creation_time: datetime
    updation_time: datetime
    messages: list[Message]

class Chunk(BaseModel):
    chunk_id:str
    conversation_id:str
    platform:str
    title:str
    text:str
    message_ids: list[str]
    time: datetime




    
    
    
    

