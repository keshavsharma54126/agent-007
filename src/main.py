import threading
import uvicorn
from src.cli.cli import app as cli_app
from src.server.server import api as fastapi_server


def start_server():
    uvicorn.run(fastapi_server, host="0.0.0.0", port=8008)


# Run server in background
threading.Thread(target=start_server, daemon=True).start()

# Run CLI in foreground
cli_app()
