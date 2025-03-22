from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/mock_api', methods=['POST'])
def mock_api():
    prompt = request.get_json()
    # Для простоты эмулируем ответ, используя последний текст из промпта
    try:
        last_message = prompt["messages"][-1]["text"]
    except (KeyError, IndexError):
        last_message = "Нет данных для ответа."
    # Формируем mock-ответ
    mock_reply = f"Mock reply: {last_message}"
    return jsonify({"text": mock_reply})

if __name__ == '__main__':
    app.run(port=5000)
