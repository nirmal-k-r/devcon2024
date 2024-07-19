import ollama

#initalisations
conversation_history = []


def chat(message):
    global conversation_history
    conversation_history.append({'role': 'user', 'content': message})
    
    stream = ollama.chat(
        model='llama3',
        messages=conversation_history,
        stream=True,
    )
    
    print("AI: ", end='', flush=True)
    response_message = ""
    for chunk in stream:
        response_message += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
    
    conversation_history.append({'role': 'assistant', 'content': response_message})

while True:
    prompt = input('\n\nYou > ')
    if prompt.lower() == 'exit':
        break
    chat(prompt)