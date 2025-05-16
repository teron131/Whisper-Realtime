import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=9090, help="Websocket port to run the server on.")
    parser.add_argument("--faster_whisper_custom_model_path", "-fw", type=str, default=None, help="Custom Faster Whisper Model")
    args = parser.parse_args()

    from whisper_live.server import TranscriptionServer

    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        port=args.port,
        faster_whisper_custom_model_path=args.faster_whisper_custom_model_path,
    )
