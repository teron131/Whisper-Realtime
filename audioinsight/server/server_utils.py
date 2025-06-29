import asyncio
import os
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

from ..logging_config import get_logger
from .server_config import CHUNK_SIZE, FFMPEG_AUDIO_PARAMS, FFPROBE_DURATION_CMD

logger = get_logger(__name__)


def get_audio_duration(file_path: str) -> float:
    """Get duration of audio file using ffprobe.

    Args:
        file_path: Path to the audio file

    Returns:
        Duration in seconds

    Raises:
        Exception: If ffprobe fails or duration is invalid
    """
    logger.info(f"Getting duration for audio file: {file_path}")

    cmd = FFPROBE_DURATION_CMD + [file_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffprobe failed: {result.stderr}")
        raise Exception(f"Failed to get audio duration: {result.stderr}")

    try:
        duration_str = result.stdout.strip()
        if not duration_str:
            raise Exception("Empty duration result from ffprobe")
        duration = float(duration_str)
        if duration <= 0:
            raise Exception(f"Invalid duration: {duration}")

        logger.info(f"Audio duration: {duration:.2f} seconds")
        return duration
    except (ValueError, IndexError) as e:
        raise Exception(f"Could not parse audio duration '{result.stdout.strip()}': {e}")


def create_temp_file(content: bytes, suffix: str) -> str:
    """Create temporary file with given content.

    Args:
        content: File content as bytes
        suffix: File suffix/extension

    Returns:
        Path to temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(content)
        temp_file.flush()
        return temp_file.name


def cleanup_temp_file(file_path: str) -> bool:
    """Clean up temporary file safely.

    Args:
        file_path: Path to file to delete

    Returns:
        True if file was deleted, False otherwise
    """
    try:
        if os.path.exists(file_path) and "/tmp/" in file_path:
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")
        return False


def setup_ffmpeg_process(file_path: str) -> subprocess.Popen:
    """Set up FFmpeg process for audio conversion.

    Args:
        file_path: Path to input audio file

    Returns:
        FFmpeg subprocess
    """
    cmd = ["ffmpeg", "-i", file_path] + FFMPEG_AUDIO_PARAMS + ["pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_audio_chunks(ffmpeg_process: subprocess.Popen) -> List[bytes]:
    """Read all audio chunks from FFmpeg process.

    Args:
        ffmpeg_process: FFmpeg subprocess

    Returns:
        List of audio chunks
    """
    chunks = []
    while True:
        chunk = ffmpeg_process.stdout.read(CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
    return chunks


def calculate_streaming_params(chunks: List[bytes], duration: float) -> Tuple[int, float, float]:
    """Calculate parameters for real-time streaming simulation.

    Args:
        chunks: List of audio chunks
        duration: Audio duration in seconds

    Returns:
        Tuple of (total_bytes, bytes_per_second, chunk_interval)

    Raises:
        Exception: If no audio data or invalid duration
    """
    total_bytes = sum(len(chunk) for chunk in chunks)
    if total_bytes == 0:
        raise Exception("No audio data received from FFmpeg")
    if duration <= 0:
        raise Exception("Invalid duration for audio file")

    bytes_per_second = total_bytes / duration
    chunk_interval = CHUNK_SIZE / bytes_per_second

    logger.info(f"Streaming {total_bytes} bytes over {duration:.2f}s " f"({bytes_per_second:.0f} bytes/s, {chunk_interval:.3f}s per chunk)")

    return total_bytes, bytes_per_second, chunk_interval


async def stream_chunks_realtime(
    chunks: List[bytes],
    chunk_interval: float,
    duration: float,
    process_func,
    progress_callback: Optional[callable] = None,
) -> float:
    """Stream audio chunks with optimized pacing for faster processing.

    Args:
        chunks: List of audio chunks to stream
        chunk_interval: Interval between chunks in seconds
        duration: Total audio duration
        process_func: Async function to process each chunk
        progress_callback: Optional callback for progress updates

    Returns:
        Total elapsed time
    """
    stream_start_time = time.time()
    progress_log_interval = max(1, int(2.0 / chunk_interval)) if chunk_interval > 0 else 1

    # OPTIMIZATION: Process 2x faster than real-time for better user experience
    # while maintaining FFmpeg stability (4x was too fast and caused restarts)
    optimized_chunk_interval = chunk_interval * 0.5  # 2x speed

    logger.info(f"Optimized streaming: {len(chunks)} chunks at {1/optimized_chunk_interval:.1f}x real-time speed")

    for i, chunk in enumerate(chunks):
        # Calculate target time for this chunk (optimized)
        target_time = stream_start_time + (i * optimized_chunk_interval)
        current_time = time.time()

        # Sleep if we're ahead of schedule (much shorter delays now)
        sleep_time = target_time - current_time
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        # Process the chunk
        await process_func(chunk)

        # Log progress periodically
        if progress_callback and i % progress_log_interval == 0:
            elapsed = time.time() - stream_start_time
            audio_progress = (i / len(chunks)) * duration
            progress_callback(audio_progress, duration, elapsed)

    # Send end of stream signal
    await process_func(b"")

    total_elapsed = time.time() - stream_start_time
    logger.info(f"Finished optimized streaming: {total_elapsed:.2f}s (target: {duration:.2f}s, speedup: {duration/total_elapsed:.1f}x)")

    return total_elapsed


def validate_file_type(content_type: str, filename: str = None) -> bool:
    """Validate if file type is allowed (very permissive - let FFmpeg handle format validation).

    Args:
        content_type: MIME type of the file
        filename: Optional filename for extension-based validation

    Returns:
        True if file type is allowed, False otherwise
    """
    # Normalize MIME types
    normalized_type = content_type.lower().strip()

    # Block obviously problematic file types
    blocked_types = {
        "text/plain",
        "text/html",
        "text/css",
        "text/javascript",
        "application/javascript",
        "application/json",
        "application/xml",
        "application/pdf",
        "application/msword",
        "application/vnd.ms-excel",
        "application/zip",
        "application/x-executable",
        "application/x-sh",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/svg+xml",
    }

    # Block obviously non-audio extensions
    if filename:
        filename_lower = filename.lower()
        blocked_extensions = {".txt", ".html", ".css", ".js", ".json", ".xml", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".sh", ".bat", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".bmp", ".tiff", ".py", ".cpp", ".java", ".php", ".rb", ".go", ".rs", ".c", ".h"}

        for ext in blocked_extensions:
            if filename_lower.endswith(ext):
                return False

    # Block the obviously problematic MIME types
    if normalized_type in blocked_types:
        return False

    # For everything else, be permissive and let FFmpeg handle it
    # This includes:
    # - All audio/* types (standard and variants)
    # - video/mp4 (for M4A files)
    # - application/octet-stream (generic binary uploads)
    # - Any other type that might be audio but misidentified
    return True


def log_progress(audio_progress: float, duration: float, elapsed: float) -> None:
    """Log streaming progress.

    Args:
        audio_progress: Current audio position in seconds
        duration: Total duration in seconds
        elapsed: Elapsed real time in seconds
    """
    logger.info(f"Streaming progress: {audio_progress:.1f}s/{duration:.1f}s " f"({elapsed:.1f}s elapsed)")
