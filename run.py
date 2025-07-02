from app import create_app
from waitress import serve

app = create_app()


@app.route('/')
def home():
    return "Hello World"

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
