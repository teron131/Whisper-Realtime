import functools
import json
import logging
import os
import time
from enum import Enum
from typing import List

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from whisper_live.backend import ServeClientFasterWhisper

logging.basicConfig(level=logging.INFO)


class ClientManager:
    def __init__(self, max_clients=4, max_connection_time=600):
        """
        Initializes the ClientManager with specified limits on client connections and connection durations.

        Args:
            max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
            max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                                 to 600 seconds (10 minutes).
        """
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time

    def add_client(self, websocket, client):
        """
        Adds a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        self.clients[websocket] = client
        self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        """
        Retrieves a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        if websocket in self.clients:
            return self.clients[websocket]
        return False

    def remove_client(self, websocket):
        """
        Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
        client if necessary.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        client = self.clients.pop(websocket, None)
        if client:
            client.cleanup()
        self.start_times.pop(websocket, None)

    def get_wait_time(self):
        """
        Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

        Returns:
            The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
        """
        wait_time = None
        for start_time in self.start_times.values():
            current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        """
        Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        if len(self.clients) >= self.max_clients:
            wait_time = self.get_wait_time()
            response = {"uid": options["uid"], "status": "WAIT", "message": wait_time}
            websocket.send(json.dumps(response))
            return True
        return False

    def is_client_timeout(self, websocket):
        """
        Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit, False otherwise.
        """
        elapsed_time = time.time() - self.start_times[websocket]
        if elapsed_time >= self.max_connection_time:
            self.clients[websocket].disconnect()
            logging.warning(f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime.")
            return True
        return False


class BackendType(Enum):
    FASTER_WHISPER = "faster_whisper"

    @staticmethod
    def valid_types() -> List[str]:
        return [BackendType.FASTER_WHISPER.value]

    @staticmethod
    def is_valid(backend: str) -> bool:
        return backend == BackendType.FASTER_WHISPER.value

    def is_faster_whisper(self) -> bool:
        return self == BackendType.FASTER_WHISPER


class TranscriptionServer:
    RATE = 16000

    def __init__(self):
        self.client_manager = None
        self.no_voice_activity_chunks = 0
        self.use_vad = True
        self.single_model = False

    def initialize_client(
        self,
        websocket,
        options,
        faster_whisper_custom_model_path,
    ):

        # Use custom model if provided
        if faster_whisper_custom_model_path and os.path.exists(faster_whisper_custom_model_path):
            logging.info(f"Using custom model {faster_whisper_custom_model_path}")
            options["model"] = faster_whisper_custom_model_path
        client = ServeClientFasterWhisper(
            websocket,
            language=options["language"],
            task=options["task"],
            client_uid=options["uid"],
            model=options["model"],
            initial_prompt=options.get("initial_prompt"),
            vad_parameters=options.get("vad_parameters"),
            use_vad=self.use_vad,
            single_model=self.single_model,
            send_last_n_segments=options.get("send_last_n_segments", 10),
            no_speech_thresh=options.get("no_speech_thresh", 0.45),
            clip_audio=options.get("clip_audio", False),
            same_output_threshold=options.get("same_output_threshold", 10),
        )
        logging.info("Running faster_whisper backend.")
        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket, faster_whisper_custom_model_path):
        try:
            logging.info("New client connected")
            options = json.loads(websocket.recv())

            if self.client_manager is None:
                max_clients = options.get("max_clients", 4)
                max_connection_time = options.get("max_connection_time", 600)
                self.client_manager = ClientManager(max_clients, max_connection_time)

            self.use_vad = options.get("use_vad")
            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False  # Indicates that the connection should not continue

            self.initialize_client(websocket, options, faster_whisper_custom_model_path)
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        """
        Retrieve the next audio frame and add it to the client's buffer.
        """
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            return False
        client.add_frames(frame_np)
        return True

    def recv_audio(self, websocket, backend: BackendType = BackendType.FASTER_WHISPER, faster_whisper_custom_model_path=None):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            backend (str): The backend to run the server with.
            faster_whisper_custom_model_path (str): path to custom faster whisper model.
        """
        self.backend = backend
        if not self.handle_new_connection(websocket, faster_whisper_custom_model_path):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def run(self, host: str, port: int = 9090, faster_whisper_custom_model_path: str = None, single_model: bool = False) -> None:
        """
        Run the transcription server using only the faster_whisper backend.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
        """
        # Validate custom model path
        if faster_whisper_custom_model_path and not os.path.exists(faster_whisper_custom_model_path):
            raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path.")
        # Configure single-model mode if requested
        if single_model:
            if faster_whisper_custom_model_path:
                logging.info("Single model mode: using custom model for all connections.")
                self.single_model = True
            else:
                logging.info("Single model mode requires a custom model path.")
        # Start the server
        handler = functools.partial(
            self.recv_audio,
            backend=BackendType.FASTER_WHISPER,
            faster_whisper_custom_model_path=faster_whisper_custom_model_path,
        )
        with serve(handler, host, port) as server:
            server.serve_forever()

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        if self.client_manager.get_client(websocket):
            self.client_manager.remove_client(websocket)
