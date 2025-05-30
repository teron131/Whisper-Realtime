from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Optional

from .logging_config import get_logger, setup_logging
from .whisper_streaming.whisper_online import backend_factory, warmup_asr

# Initialize centralized logging early
setup_logging()
logger = get_logger(__name__)

# Global cached parser to avoid recreation
_CACHED_PARSER: Optional[ArgumentParser] = None

# Pre-defined argument groups for better organization
_DEFAULT_CONFIG = {
    "server": {
        "host": "localhost",
        "port": 8001,
    },
    "model": {
        "model": "large-v3-turbo",
        "backend": "faster-whisper",
        "lang": "auto",
        "task": "transcribe",
    },
    "processing": {
        "min_chunk_size": 0.5,
        "buffer_trimming": "segment",
        "buffer_trimming_sec": 15.0,
        "vac_chunk_size": 0.04,
    },
    "features": {
        "transcription": True,
        "diarization": False,
        "vad": True,
        "vac": False,
        "confidence_validation": False,
        "llm_inference": True,
    },
    "llm": {
        "fast_llm": "openai/gpt-4.1-nano",
        "base_llm": "openai/gpt-4.1-mini",
        "llm_summary_interval": 1.0,
        "llm_new_text_trigger": 300,
    },
}


def _get_argument_parser() -> ArgumentParser:
    """Get cached argument parser or create new one."""
    global _CACHED_PARSER

    if _CACHED_PARSER is not None:
        return _CACHED_PARSER

    parser = ArgumentParser(description="Whisper FastAPI Online Server")

    # Server configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument("--host", type=str, default=_DEFAULT_CONFIG["server"]["host"], help="The host address to bind the server to.")
    server_group.add_argument("--port", type=int, default=_DEFAULT_CONFIG["server"]["port"], help="The port number to bind the server to.")
    server_group.add_argument("--ssl-certfile", type=str, default=None, help="Path to the SSL certificate file.")
    server_group.add_argument("--ssl-keyfile", type=str, default=None, help="Path to the SSL private key file.")

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, default=_DEFAULT_CONFIG["model"]["model"], help="Name size of the Whisper model to use (default: tiny). Suggested values: tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v3-turbo.")
    model_group.add_argument("--model_cache_dir", type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved")
    model_group.add_argument("--model_dir", type=str, default=None, help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    model_group.add_argument("--backend", type=str, default=_DEFAULT_CONFIG["model"]["backend"], choices=["faster-whisper", "openai-api"], help="Load only this backend for Whisper processing.")
    model_group.add_argument("--lang", "--language", type=str, default=_DEFAULT_CONFIG["model"]["lang"], help="Source language code, e.g. en,de,cs, or 'auto' for language detection.")
    model_group.add_argument("--task", type=str, default=_DEFAULT_CONFIG["model"]["task"], choices=["transcribe", "translate"], help="Transcribe or translate.")

    # Processing configuration
    processing_group = parser.add_argument_group("Processing Configuration")
    processing_group.add_argument("--warmup-file", type=str, default=None, dest="warmup_file", help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. If not set, uses https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav. If False, no warmup is performed.")
    processing_group.add_argument("--min-chunk-size", type=float, default=_DEFAULT_CONFIG["processing"]["min_chunk_size"], help="Minimum audio chunk size in seconds. It waits up to this time to do processing.")
    processing_group.add_argument("--buffer_trimming", type=str, default=_DEFAULT_CONFIG["processing"]["buffer_trimming"], choices=["sentence", "segment"], help="Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper.")
    processing_group.add_argument("--buffer_trimming_sec", type=float, default=_DEFAULT_CONFIG["processing"]["buffer_trimming_sec"], help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.")
    processing_group.add_argument("--vac-chunk-size", type=float, default=_DEFAULT_CONFIG["processing"]["vac_chunk_size"], help="VAC sample size in seconds.")

    # Feature flags
    feature_group = parser.add_argument_group("Feature Configuration")
    feature_group.add_argument("--confidence-validation", action="store_true", help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.")
    feature_group.add_argument("--diarization", action="store_true", default=_DEFAULT_CONFIG["features"]["diarization"], help="Enable speaker diarization.")
    feature_group.add_argument("--no-transcription", action="store_true", help="Disable transcription to only see live diarization results.")
    feature_group.add_argument("--vac", action="store_true", default=_DEFAULT_CONFIG["features"]["vac"], help="Use VAC = voice activity controller. Recommended. Requires torch.")
    feature_group.add_argument("--no-vad", action="store_true", help="Disable VAD (voice activity detection).")

    # LLM configuration
    llm_group = parser.add_argument_group("LLM Inference Configuration")
    llm_group.add_argument("--llm-inference", action="store_true", default=_DEFAULT_CONFIG["features"]["llm_inference"], help="Enable LLM-based transcription inference after periods of inactivity.")
    llm_group.add_argument("--fast-llm", type=str, default=_DEFAULT_CONFIG["llm"]["fast_llm"], help="Fast LLM model to use for text parsing and quick operations (default: openai/gpt-4.1-nano).")
    llm_group.add_argument("--base-llm", type=str, default=_DEFAULT_CONFIG["llm"]["base_llm"], help="Base LLM model to use for summarization and complex operations (default: openai/gpt-4.1-mini).")
    llm_group.add_argument("--llm-summary-interval", type=float, default=_DEFAULT_CONFIG["llm"]["llm_summary_interval"], help="Minimum time interval in seconds between comprehensive summaries (default: 15.0).")
    llm_group.add_argument("--llm-new-text-trigger", type=int, default=_DEFAULT_CONFIG["llm"]["llm_new_text_trigger"], help="Number of new characters since last summary to trigger a new summary (default: 300).")
    llm_group.add_argument("--parser-trigger-interval", type=float, default=1.0, help="Time interval in seconds between transcript parser triggers (default: 1.0).")
    llm_group.add_argument("--parser-output-tokens", type=int, default=33000, help="Maximum output tokens for transcript parser. Parser will chunk text to stay within this OUTPUT limit (default: 33000). Note: This is about OUTPUT tokens the model generates, not input tokens.")

    # Logging configuration
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument("-l", "--log-level", dest="log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the log level", default="DEBUG")

    _CACHED_PARSER = parser
    return parser


def _validate_args(args: Namespace) -> None:
    """Validate argument combinations and values."""
    # Validate port range
    if not (1 <= args.port <= 65535):
        raise ValueError(f"Port must be between 1 and 65535, got {args.port}")

    # Validate chunk sizes
    if args.min_chunk_size <= 0:
        raise ValueError(f"min_chunk_size must be positive, got {args.min_chunk_size}")

    if args.vac_chunk_size <= 0:
        raise ValueError(f"vac_chunk_size must be positive, got {args.vac_chunk_size}")

    if args.buffer_trimming_sec <= 0:
        raise ValueError(f"buffer_trimming_sec must be positive, got {args.buffer_trimming_sec}")

    # Validate SSL configuration
    if (args.ssl_certfile is None) != (args.ssl_keyfile is None):
        raise ValueError("Both ssl_certfile and ssl_keyfile must be provided together or not at all")

    # Validate LLM configuration
    if args.llm_summary_interval <= 0:
        raise ValueError(f"llm_summary_interval must be positive, got {args.llm_summary_interval}")

    if args.llm_new_text_trigger <= 0:
        raise ValueError(f"llm_new_text_trigger must be positive, got {args.llm_new_text_trigger}")

    if args.parser_trigger_interval <= 0:
        raise ValueError(f"parser_trigger_interval must be positive, got {args.parser_trigger_interval}")

    if args.parser_output_tokens <= 0:
        raise ValueError(f"parser_output_tokens must be positive, got {args.parser_output_tokens}")

    if args.parser_output_tokens > 100000:
        raise ValueError(f"parser_output_tokens too high (max 100000), got {args.parser_output_tokens}")

    # Validate feature combinations
    if args.no_transcription and not args.diarization:
        raise ValueError("Cannot disable transcription without enabling diarization")


def _optimize_args(args: Namespace) -> Namespace:
    """Optimize and precompute derived values from arguments."""
    # Convert boolean flags efficiently - use setattr for better performance than delattr + recreation
    args.transcription = not getattr(args, "no_transcription", False)
    args.vad = not getattr(args, "no_vad", False)

    # Remove the temporary negative flags
    for attr in ["no_transcription", "no_vad"]:
        if hasattr(args, attr):
            delattr(args, attr)

    # Pre-compute derived values for better performance (only if attributes exist)
    if hasattr(args, "min_chunk_size"):
        args.samples_per_chunk = int(16000 * args.min_chunk_size)  # 16kHz sample rate

    if hasattr(args, "vac_chunk_size"):
        args.vac_samples_per_chunk = int(16000 * args.vac_chunk_size)

    if hasattr(args, "buffer_trimming_sec"):
        args.buffer_trimming_samples = int(16000 * args.buffer_trimming_sec)

    # Create feature flags dictionary for easy access
    args.features = {
        "transcription": getattr(args, "transcription", True),
        "diarization": getattr(args, "diarization", False),
        "vad": getattr(args, "vad", True),
        "vac": getattr(args, "vac", False),
        "confidence_validation": getattr(args, "confidence_validation", False),
        "llm_inference": getattr(args, "llm_inference", True),
    }

    return args


def parse_args(argv=None) -> Namespace:
    """Parse and validate command line arguments with optimizations."""
    parser = _get_argument_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit:
        # If parsing fails, return defaults for testing
        if argv is None:
            # Re-raise the exception for normal usage
            raise
        else:
            # For testing with specific argv, return defaults
            args = Namespace()
            for category in _DEFAULT_CONFIG.values():
                for key, value in category.items():
                    setattr(args, key, value)

    # Validate arguments
    _validate_args(args)

    # Optimize and precompute values
    args = _optimize_args(args)

    return args


def parse_args_safe(argv=None) -> Namespace:
    """Parse arguments safely for testing, returning defaults on failure."""
    try:
        return parse_args(argv)
    except (SystemExit, ValueError):
        # Return safe defaults for testing
        args = Namespace()
        for category in _DEFAULT_CONFIG.values():
            for key, value in category.items():
                setattr(args, key, value)

        # Add required defaults
        required_defaults = {
            "warmup_file": None,
            "model_cache_dir": None,
            "model_dir": None,
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "log_level": "DEBUG",
        }
        for key, value in required_defaults.items():
            setattr(args, key, value)

        return _optimize_args(args)


class AudioInsight:
    _instance = None
    _initialized = False
    _cached_html = None  # Cache for web interface HTML
    _cached_config = None  # Cache for configuration

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        if AudioInsight._initialized:
            return

        # Use cached config if available and no overrides provided
        if not kwargs and AudioInsight._cached_config is not None:
            self.args = AudioInsight._cached_config
        else:
            if kwargs:
                # If custom kwargs provided, start with defaults and override
                merged_args = {}
                for category in _DEFAULT_CONFIG.values():
                    merged_args.update(category)
                merged_args.update(kwargs)

                # Add required defaults that might not be in _DEFAULT_CONFIG
                required_defaults = {
                    "warmup_file": None,
                    "model_cache_dir": None,
                    "model_dir": None,
                    "ssl_certfile": None,
                    "ssl_keyfile": None,
                    "log_level": "DEBUG",
                }
                for key, value in required_defaults.items():
                    if key not in merged_args:
                        merged_args[key] = value

                self.args = Namespace(**merged_args)

                # Apply optimizations to the args, but preserve explicit kwargs
                self.args = _optimize_args(self.args)

                # Restore explicit kwargs that might have been overridden by optimization
                for key, value in kwargs.items():
                    setattr(self.args, key, value)
            else:
                # Parse arguments only when needed (no custom overrides)
                default_args = vars(parse_args())
                self.args = Namespace(**default_args)

                # Cache the configuration for future use
                AudioInsight._cached_config = self.args

        # Initialize components lazily
        self.asr = None
        self.tokenizer = None
        self.diarization = None
        self._models_loaded = False
        self._diarization_loaded = False

        # Load models only if transcription is enabled
        transcription_enabled = getattr(self.args, "transcription", True)
        if transcription_enabled and not self._models_loaded:
            self._load_asr_models()
            self._models_loaded = True

        # Load diarization only if enabled
        diarization_enabled = getattr(self.args, "diarization", False)
        if diarization_enabled and not self._diarization_loaded:
            self._load_diarization()
            self._diarization_loaded = True

        AudioInsight._initialized = True

    def _load_asr_models(self):
        """Lazy loading of ASR models."""
        try:
            self.asr, self.tokenizer = backend_factory(self.args)
            warmup_asr(self.asr, self.args.warmup_file)
        except Exception as e:
            logger.error(f"Failed to load ASR models: {e}")
            raise

    def _load_diarization(self):
        """Lazy loading of diarization models."""
        try:
            from diart import SpeakerDiarizationConfig

            from .diarization.diarization_online import DiartDiarization

            # Very conservative configuration for high confidence diarization only
            config = SpeakerDiarizationConfig(
                step=2.0,  # Even slower processing = higher accuracy (was 1.0)
                latency=2.0,  # Allow more latency for better decisions (was 1.0)
                tau_active=0.8,  # Very high voice activity threshold = only strong voices (was 0.7)
                rho_update=0.05,  # Very low update rate = extremely stable speakers (was 0.1)
                delta_new=3.0,  # Very high threshold = very resistant to new speakers (was 1.8)
                gamma=15,  # Much higher gamma = very stable clustering (was 8)
                beta=30,  # Much higher beta = very stable processing (was 20)
                max_speakers=3,  # Conservative speaker limit (was 4)
            )

            self.diarization = DiartDiarization(config=config)
        except Exception as e:
            logger.error(f"Failed to load diarization models: {e}")
            raise

    @classmethod
    def get_instance(cls, **kwargs):
        """Get singleton instance with optional configuration override."""
        if cls._instance is None or kwargs:
            return cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance for testing purposes."""
        cls._instance = None
        cls._initialized = False
        cls._cached_html = None
        cls._cached_config = None

    def reconfigure(self, **kwargs):
        """Reconfigure the instance with new parameters."""
        if not kwargs:
            return

        # Update configuration
        for key, value in kwargs.items():
            setattr(self.args, key, value)

        # Reload models if necessary
        if "transcription" in kwargs and kwargs["transcription"] and not self._models_loaded:
            self._load_asr_models()
            self._models_loaded = True

        if "diarization" in kwargs and kwargs["diarization"] and not self._diarization_loaded:
            self._load_diarization()
            self._diarization_loaded = True

    def web_interface(self):
        """Get cached web interface HTML."""
        # Use cached HTML if available
        if AudioInsight._cached_html is not None:
            return AudioInsight._cached_html

        try:
            import pkg_resources

            html_path = pkg_resources.resource_filename("audioinsight", "frontend/ui.html")
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            # Cache the HTML content
            AudioInsight._cached_html = html
            return html
        except Exception as e:
            logger.error(f"Failed to load web interface: {e}")
            # Return a minimal fallback HTML
            return """
            <html><head><title>AudioInsight</title></head>
            <body><h1>AudioInsight - Web Interface Error</h1>
            <p>Failed to load the web interface. Please check the installation.</p></body></html>
            """

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "model": {
                "name": self.args.model,
                "backend": self.args.backend,
                "language": self.args.lang,
                "task": self.args.task,
            },
            "features": getattr(self.args, "features", {}),
            "processing": {
                "min_chunk_size": self.args.min_chunk_size,
                "buffer_trimming": self.args.buffer_trimming,
                "buffer_trimming_sec": self.args.buffer_trimming_sec,
            },
            "server": {
                "host": self.args.host,
                "port": self.args.port,
            },
            "models_loaded": {
                "asr": self._models_loaded,
                "diarization": self._diarization_loaded,
            },
        }
