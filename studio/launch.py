import threading
import webbrowser

import uvicorn


def open_browser():
    webbrowser.open("http://127.0.0.1:8765")


if __name__ == "__main__":
    threading.Timer(1.3, open_browser).start()
    uvicorn.run("studio.main:app", host="127.0.0.1", port=8765, reload=False)

