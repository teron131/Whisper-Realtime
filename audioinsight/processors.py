import asyncio
import math
import re
import traceback
from datetime import timedelta
from time import sleep, time
from typing import Optional

import ffmpeg
import numpy as np
import opencc

from .llm import LLMSummarizer, LLMTrigger, ParsedTranscript, Parser, ParserConfig
from .logging_config import get_logger
from .main import AudioInsight
from .timed_objects import ASRToken
from .whisper_streaming.whisper_online import online_factory

# Initialize logging using centralized configuration
logger = get_logger(__name__)

SENTINEL = object()  # unique sentinel object for end of stream marker

# Cache OpenCC converter instance to avoid recreation
_s2hk_converter = None

# Pre-compile regex for sentence splitting
_sentence_split_regex = re.compile(r"[.!?]+")

# Cache timedelta formatting for common values
_cached_timedeltas = {}


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS with caching for performance."""
    if seconds == 0:
        return "0:00:00"

    int_seconds = int(seconds)
    if int_seconds in _cached_timedeltas:
        return _cached_timedeltas[int_seconds]

    result = str(timedelta(seconds=int_seconds))

    # Cache up to 3600 entries (1 hour worth of seconds)
    if len(_cached_timedeltas) < 3600:
        _cached_timedeltas[int_seconds] = result

    return result


def s2hk(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese with cached converter."""
    if not text:
        return text

    global _s2hk_converter
    if _s2hk_converter is None:
        _s2hk_converter = opencc.OpenCC("s2hk")

    return _s2hk_converter.convert(text)


class FFmpegProcessor:
    """Handles FFmpeg process management and audio data conversion."""

    def __init__(self, args):
        self.args = args
        self.sample_rate = 16000
        self.channels = 1
        self.bytes_per_sample = 2

        # Pre-compute commonly used values
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.sample_rate_str = str(self.sample_rate)  # Cache string conversion

        # Pre-allocate buffers for better memory efficiency
        self.max_buffer_size = self.max_bytes_per_sec * 2  # Double buffer size for safety
        self.pcm_buffer = bytearray(self.max_buffer_size)
        self.pcm_buffer_length = 0  # Track actual data length

        # Pre-allocate numpy arrays to avoid repeated allocation
        self._temp_int16_array = np.empty(self.max_buffer_size // 2, dtype=np.int16)
        self._temp_float32_array = np.empty(self.max_buffer_size // 2, dtype=np.float32)

        # Timing settings
        self.last_ffmpeg_activity = time()
        self.ffmpeg_health_check_interval = 10  # Less frequent health checks
        self.ffmpeg_max_idle_time = 30  # Allow much longer idle time before restart

        # FFmpeg process
        self.ffmpeg_process = self.start_ffmpeg_decoder()

    def convert_pcm_to_float(self, pcm_data, length=None):
        """Convert PCM buffer in s16le format to normalized NumPy array with pre-allocated buffers."""
        if length is None:
            length = len(pcm_data)

        # Use pre-allocated arrays for better performance
        num_samples = length // 2
        if num_samples > len(self._temp_int16_array):
            # Resize if needed (rare case)
            self._temp_int16_array = np.empty(num_samples, dtype=np.int16)
            self._temp_float32_array = np.empty(num_samples, dtype=np.float32)

        # Copy data into pre-allocated buffer
        self._temp_int16_array[:num_samples] = np.frombuffer(pcm_data[:length], dtype=np.int16)

        # Convert to float32 in-place
        np.divide(self._temp_int16_array[:num_samples], 32768.0, out=self._temp_float32_array[:num_samples])

        # Return a copy of the needed portion
        return self._temp_float32_array[:num_samples].copy()

    def append_to_pcm_buffer(self, chunk):
        """Efficiently append audio chunk to PCM buffer."""
        chunk_len = len(chunk)
        new_length = self.pcm_buffer_length + chunk_len

        # Resize buffer if needed
        if new_length > len(self.pcm_buffer):
            new_size = max(new_length, len(self.pcm_buffer) * 2)
            new_buffer = bytearray(new_size)
            new_buffer[: self.pcm_buffer_length] = self.pcm_buffer[: self.pcm_buffer_length]
            self.pcm_buffer = new_buffer

        # Append new data
        self.pcm_buffer[self.pcm_buffer_length : new_length] = chunk
        self.pcm_buffer_length = new_length

    def get_pcm_data(self, max_bytes):
        """Get PCM data up to max_bytes and remove it from buffer."""
        actual_bytes = min(self.pcm_buffer_length, max_bytes)

        # Get data
        data = bytes(self.pcm_buffer[:actual_bytes])

        # Shift remaining data to front
        if actual_bytes < self.pcm_buffer_length:
            remaining = self.pcm_buffer_length - actual_bytes
            self.pcm_buffer[:remaining] = self.pcm_buffer[actual_bytes : self.pcm_buffer_length]
            self.pcm_buffer_length = remaining
        else:
            self.pcm_buffer_length = 0

        return data

    def start_ffmpeg_decoder(self):
        """Start FFmpeg process for WebM to PCM conversion."""
        try:
            # Close any existing stdout/stderr before creating new process
            if hasattr(self, "ffmpeg_process") and self.ffmpeg_process:
                try:
                    if self.ffmpeg_process.stdout and not self.ffmpeg_process.stdout.closed:
                        self.ffmpeg_process.stdout.close()
                    if self.ffmpeg_process.stderr and not self.ffmpeg_process.stderr.closed:
                        self.ffmpeg_process.stderr.close()
                except:
                    pass

            # Create new FFmpeg process with explicit error handling
            process = ffmpeg.input("pipe:0", format="webm").output("pipe:1", format="s16le", acodec="pcm_s16le", ac=self.channels, ar=self.sample_rate_str).run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)  # Suppress FFmpeg output to avoid buffer issues

            # Verify the process started correctly
            if process.poll() is not None:
                raise RuntimeError("FFmpeg process failed to start")

            return process

        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            raise

    async def restart_ffmpeg(self):
        """Restart the FFmpeg process after failure."""
        logger.warning("Restarting FFmpeg process...")

        # Force cleanup of existing process more aggressively
        if self.ffmpeg_process:
            try:
                # Close stdin immediately to signal shutdown
                if self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                    try:
                        self.ffmpeg_process.stdin.close()
                    except:
                        pass

                # Close stdout and stderr
                if self.ffmpeg_process.stdout and not self.ffmpeg_process.stdout.closed:
                    try:
                        self.ffmpeg_process.stdout.close()
                    except:
                        pass

                if self.ffmpeg_process.stderr and not self.ffmpeg_process.stderr.closed:
                    try:
                        self.ffmpeg_process.stderr.close()
                    except:
                        pass

                # Terminate the process if it's still running
                if self.ffmpeg_process.poll() is None:
                    logger.info("Terminating existing FFmpeg process")
                    self.ffmpeg_process.terminate()

                    # Wait for termination with timeout
                    try:
                        await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait), timeout=3.0)
                    except asyncio.TimeoutError:
                        logger.warning("FFmpeg process did not terminate, killing forcefully")
                        self.ffmpeg_process.kill()
                        try:
                            await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait), timeout=2.0)
                        except asyncio.TimeoutError:
                            logger.error("FFmpeg process could not be killed")

            except Exception as e:
                logger.error(f"Error during FFmpeg process termination: {e}")

        # Clear the reference
        self.ffmpeg_process = None

        # Wait a moment before restarting to ensure cleanup
        await asyncio.sleep(0.5)

        # Start new process with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting new FFmpeg process (attempt {attempt + 1}/{max_retries})")
                self.ffmpeg_process = self.start_ffmpeg_decoder()
                self.pcm_buffer_length = 0  # Reset buffer length for new process
                self.last_ffmpeg_activity = time()
                logger.info("FFmpeg process restarted successfully")
                return

            except Exception as e:
                logger.error(f"Failed to restart FFmpeg process (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    # Wait longer between retries
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.error("Maximum retries reached for FFmpeg process - continuing without restart")
                    return

        logger.warning("All FFmpeg processing attempts failed but continuing")
        return

    async def read_audio_data(self, transcription_queue=None, diarization_queue=None):
        """Read audio data from FFmpeg stdout and process it."""
        loop = asyncio.get_event_loop()
        beg = time()

        while True:
            try:
                current_time = time()
                elapsed_time = math.floor((current_time - beg) * 10) / 10
                buffer_size = max(int(32000 * elapsed_time), 4096)
                beg = current_time

                # Detect idle state much more conservatively
                if current_time - self.last_ffmpeg_activity > 60.0:  # Much longer idle threshold
                    logger.warning(f"FFmpeg process idle for {current_time - self.last_ffmpeg_activity:.2f}s. Restarting...")
                    await self.restart_ffmpeg()
                    beg = time()
                    self.last_ffmpeg_activity = time()
                    continue

                chunk = await loop.run_in_executor(None, self.ffmpeg_process.stdout.read, buffer_size)
                if chunk:
                    self.last_ffmpeg_activity = time()

                if not chunk:
                    logger.info("FFmpeg stdout closed, no more data to read.")
                    break

                self.append_to_pcm_buffer(chunk)

                # OPTIMIZATION: Increase minimum buffer for transcription to reduce processing frequency
                # Use larger buffer requirement to avoid VAD processing tiny chunks
                min_transcription_buffer = max(self.bytes_per_sec // 2, 8192)  # 0.5 seconds minimum (increased from 0.25s)
                if self.pcm_buffer_length >= min_transcription_buffer:
                    # Only log processing every 20 seconds and make it more concise
                    if not hasattr(self, "_ffmpeg_log_counter"):
                        self._ffmpeg_log_counter = 0
                        self._last_processing_log = 0
                    self._ffmpeg_log_counter += 1

                    current_log_time = time()
                    if current_log_time - self._last_processing_log > 20.0:  # Log every 20 seconds, less frequently
                        logger.info(f"🎵 Processing: {self._ffmpeg_log_counter} chunks, {self.pcm_buffer_length / self.bytes_per_sec:.1f}s buffered")
                        self._last_processing_log = current_log_time

                    if self.pcm_buffer_length > self.max_bytes_per_sec:
                        logger.warning(f"Audio buffer large: {self.pcm_buffer_length / self.bytes_per_sec:.2f}s")

                    # Process audio chunk - take larger chunks to reduce Whisper VAD calls
                    bytes_to_process = min(self.pcm_buffer_length, self.max_bytes_per_sec)
                    pcm_array = self.convert_pcm_to_float(self.get_pcm_data(bytes_to_process))

                    # Send to transcription if enabled
                    if self.args.transcription and transcription_queue:
                        await transcription_queue.put(pcm_array.copy())

                    # Send to diarization if enabled - use larger chunks less frequently to prevent backlog
                    if self.args.diarization and diarization_queue:
                        # Only send to diarization every 2 seconds worth of audio to prevent queue overload
                        if not hasattr(self, "_diarization_chunk_buffer"):
                            self._diarization_chunk_buffer = []
                            self._diarization_buffer_size = 0

                        self._diarization_chunk_buffer.append(pcm_array.copy())
                        self._diarization_buffer_size += len(pcm_array)

                        # Send larger chunks (2 seconds worth) to diarization to reduce processing frequency
                        diarization_chunk_threshold = self.bytes_per_sec * 2  # 2 seconds of audio
                        if self._diarization_buffer_size >= diarization_chunk_threshold:
                            # Check queue size to prevent overload
                            if diarization_queue.qsize() < 5:  # Limit queue to 5 items (10 seconds of audio)
                                # Concatenate buffered chunks
                                combined_audio = np.concatenate(self._diarization_chunk_buffer)
                                await diarization_queue.put(combined_audio)
                                # Remove verbose diarization logging entirely - only log errors
                            else:
                                # Only warn occasionally about queue being full
                                if not hasattr(self, "_last_queue_warning"):
                                    self._last_queue_warning = 0
                                if time() - self._last_queue_warning > 10.0:  # Warn every 10 seconds
                                    logger.warning(f"Diarization queue full ({diarization_queue.qsize()} items)")
                                    self._last_queue_warning = time()

                            # Reset buffer regardless of whether we sent data
                            self._diarization_chunk_buffer = []
                            self._diarization_buffer_size = 0

                    # Sleep if no processing is happening
                    if not self.args.transcription and not self.args.diarization:
                        await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        logger.info("FFmpeg processing finished. Signaling downstream processors.")
        if self.args.transcription and transcription_queue:
            await transcription_queue.put(SENTINEL)
        if self.args.diarization and diarization_queue:
            await diarization_queue.put(SENTINEL)

    async def process_audio_chunk(self, message):
        """Process incoming audio data."""
        retry_count = 0
        max_retries = 1  # Reduce max retries to prevent restart loops

        while retry_count < max_retries:
            try:
                # Check if FFmpeg process is available and healthy
                if not self.ffmpeg_process or not hasattr(self.ffmpeg_process, "stdin") or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process not available, restarting...")
                    await self.restart_ffmpeg()
                    retry_count += 1
                    continue

                # Check if stdin is still open
                if self.ffmpeg_process.stdin.closed:
                    logger.warning("FFmpeg stdin is closed, restarting...")
                    await self.restart_ffmpeg()
                    retry_count += 1
                    continue

                loop = asyncio.get_running_loop()

                # Write operation with more generous timeout to prevent restarts
                write_timeout = 10.0 if retry_count > 0 else 8.0  # Much more generous timeouts
                try:
                    await asyncio.wait_for(loop.run_in_executor(None, lambda: self.ffmpeg_process.stdin.write(message)), timeout=write_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"FFmpeg write timeout ({write_timeout}s) - may be processing heavy load")
                    # Don't restart immediately on timeout - just log and continue
                    self.last_ffmpeg_activity = time()
                    return

                # Flush operation with more generous timeout
                flush_timeout = 6.0 if retry_count > 0 else 4.0  # Much more generous timeouts
                try:
                    await asyncio.wait_for(loop.run_in_executor(None, self.ffmpeg_process.stdin.flush), timeout=flush_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"FFmpeg flush timeout ({flush_timeout}s) - may be processing heavy load")
                    # Don't restart immediately on timeout - just log and continue
                    self.last_ffmpeg_activity = time()
                    return

                # Success - update activity time and return
                self.last_ffmpeg_activity = time()
                return

            except (BrokenPipeError, AttributeError, OSError) as e:
                retry_count += 1
                logger.warning(f"FFmpeg error: {e}. Retry {retry_count}/{max_retries}")

                if retry_count < max_retries:
                    await self.restart_ffmpeg()
                    await asyncio.sleep(1.0)  # Longer wait between retries
                else:
                    logger.error("Maximum retries reached for FFmpeg process - continuing without restart")
                    return

        logger.warning("All FFmpeg processing attempts failed but continuing")
        return

    def cleanup(self):
        """Clean up FFmpeg resources."""
        logger.info("Starting FFmpeg cleanup...")

        if self.ffmpeg_process:
            try:
                # Close all file descriptors first
                if self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                    try:
                        self.ffmpeg_process.stdin.close()
                        logger.debug("FFmpeg stdin closed")
                    except Exception as e:
                        logger.warning(f"Error closing ffmpeg stdin during cleanup: {e}")

                if self.ffmpeg_process.stdout and not self.ffmpeg_process.stdout.closed:
                    try:
                        self.ffmpeg_process.stdout.close()
                        logger.debug("FFmpeg stdout closed")
                    except Exception as e:
                        logger.warning(f"Error closing ffmpeg stdout during cleanup: {e}")

                if self.ffmpeg_process.stderr and not self.ffmpeg_process.stderr.closed:
                    try:
                        self.ffmpeg_process.stderr.close()
                        logger.debug("FFmpeg stderr closed")
                    except Exception as e:
                        logger.warning(f"Error closing ffmpeg stderr during cleanup: {e}")

                # Terminate the process if it's still running
                if self.ffmpeg_process.poll() is None:
                    logger.info("Terminating FFmpeg process during cleanup...")
                    try:
                        self.ffmpeg_process.terminate()
                        # Wait for termination with timeout
                        try:
                            self.ffmpeg_process.wait(timeout=3.0)
                            logger.debug("FFmpeg process terminated gracefully")
                        except:  # subprocess.TimeoutExpired and other exceptions
                            logger.warning("FFmpeg did not terminate gracefully, killing forcefully")
                            self.ffmpeg_process.kill()
                            try:
                                self.ffmpeg_process.wait(timeout=2.0)
                                logger.debug("FFmpeg process killed successfully")
                            except:
                                logger.error("FFmpeg process could not be killed")
                    except Exception as e:
                        logger.warning(f"Error terminating FFmpeg process during cleanup: {e}")
                else:
                    logger.debug("FFmpeg process already terminated")

            except Exception as e:
                logger.error(f"Error during FFmpeg process cleanup: {e}")
            finally:
                # Always clear the reference
                self.ffmpeg_process = None

        # Clear all memory buffers aggressively
        if hasattr(self, "pcm_buffer"):
            self.pcm_buffer = bytearray(self.max_buffer_size)
            self.pcm_buffer_length = 0

        # Clear pre-allocated arrays
        if hasattr(self, "_temp_int16_array"):
            self._temp_int16_array.fill(0)
        if hasattr(self, "_temp_float32_array"):
            self._temp_float32_array.fill(0)

        # Clear any diarization buffers
        if hasattr(self, "_diarization_chunk_buffer"):
            self._diarization_chunk_buffer.clear()
            self._diarization_buffer_size = 0

        # Reset timing
        self.last_ffmpeg_activity = time()

        logger.info("FFmpeg cleanup completed successfully")


class TranscriptionProcessor:
    """Handles speech-to-text transcription processing."""

    def __init__(self, args, asr, tokenizer, coordinator=None):
        self.args = args
        self.coordinator = coordinator  # Reference to AudioProcessor for timing
        self.online = None
        self.full_transcription = ""
        self.sep = " "  # Default separator

        # Parser timing control
        self.last_parser_trigger_time = 0.0
        self.accumulated_text_for_parser = ""

        # Initialize transcription engine if enabled
        if self.args.transcription:
            self.online = online_factory(self.args, asr, tokenizer)
            self.sep = self.online.asr.sep

    async def process(self, transcription_queue, update_callback, llm=None):
        """Process audio chunks for transcription."""
        self.full_transcription = ""
        if self.online:
            self.sep = self.online.asr.sep

        logger.info("🎙️ Transcription processor started")

        while True:
            try:
                pcm_array = await transcription_queue.get()
                if pcm_array is SENTINEL:
                    transcription_queue.task_done()
                    break

                # OPTIMIZATION: Reduce logging frequency to prevent spam
                if not hasattr(self, "_transcription_log_counter"):
                    self._transcription_log_counter = 0
                    self._last_chunk_log = 0
                self._transcription_log_counter += 1

                # Log every 120 seconds instead of 60 seconds for less spam
                current_log_time = time()
                if current_log_time - self._last_chunk_log > 120.0:
                    logger.info(f"🎵 Transcription: {self._transcription_log_counter} chunks processed")
                    self._last_chunk_log = current_log_time

                if not self.online:  # Should not happen if queue is used
                    logger.warning("Transcription processor: self.online not initialized.")
                    transcription_queue.task_done()
                    continue

                asr_internal_buffer_duration_s = len(self.online.audio_buffer) / self.online.SAMPLING_RATE

                # Calculate timing like the original - use coordinator references
                transcription_lag_s = 0.0
                if self.coordinator:
                    transcription_lag_s = max(0.0, time() - self.coordinator.beg_loop - self.coordinator.end_buffer)

                # OPTIMIZATION: Reduce ASR processing logs significantly
                if not hasattr(self, "_last_asr_log_time"):
                    self._last_asr_log_time = 0
                current_time = time()
                if current_time - self._last_asr_log_time > 60.0:  # Log every 60 seconds (increased from 20s)
                    logger.info(f"ASR: buffer={asr_internal_buffer_duration_s:.1f}s, lag={transcription_lag_s:.1f}s")
                    self._last_asr_log_time = current_time

                # Process transcription
                self.online.insert_audio_chunk(pcm_array)
                new_tokens = self.online.process_iter()

                if new_tokens:
                    self.full_transcription += self.sep.join([t.text for t in new_tokens])
                    # Remove token generation logging entirely - too verbose

                # Get buffer information
                _buffer = self.online.get_buffer()
                buffer = _buffer.text
                end_buffer = _buffer.end if _buffer.end else (new_tokens[-1].end if new_tokens else 0)

                # Avoid duplicating content
                if buffer in self.full_transcription:
                    buffer = ""

                await update_callback(new_tokens, buffer, end_buffer, self.full_transcription, self.sep)

                # Update LLM inference processor asynchronously (non-blocking)
                if llm and new_tokens:
                    new_text = self.sep.join([t.text for t in new_tokens])
                    # Convert to Traditional Chinese for consistency
                    new_text_converted = s2hk(new_text) if new_text else new_text

                    # Accumulate text for parser with timing control
                    if self.coordinator and new_text_converted.strip():
                        # Accumulate text for parser
                        if self.accumulated_text_for_parser:
                            self.accumulated_text_for_parser += self.sep + new_text_converted
                        else:
                            self.accumulated_text_for_parser = new_text_converted

                        # Check if it's time to trigger parser (every second)
                        current_time = time()
                        parser_interval = getattr(self.coordinator.transcript_parser.config, "trigger_interval_seconds", 1.0) if self.coordinator.transcript_parser else 1.0

                        if current_time - self.last_parser_trigger_time >= parser_interval:
                            # Get accumulated speaker info (use most recent speaker)
                            speaker_info = None
                            if new_tokens and hasattr(new_tokens[0], "speaker") and new_tokens[0].speaker is not None:
                                speaker_info = {"speaker": new_tokens[0].speaker}

                            # Send accumulated text to parser
                            asyncio.create_task(self._update_coordinator_parser_async(self.coordinator, self.accumulated_text_for_parser, speaker_info))

                            # Reset accumulation
                            self.accumulated_text_for_parser = ""
                            self.last_parser_trigger_time = current_time

                transcription_queue.task_done()

            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                if "pcm_array" in locals() and pcm_array is not SENTINEL:  # Check if pcm_array was assigned from queue
                    transcription_queue.task_done()
        logger.info("Transcription processor finished.")

    async def _update_coordinator_parser_async(self, coordinator, text, speaker_info):
        """Asynchronously update coordinator's transcript parser without blocking transcription."""
        try:
            if coordinator and hasattr(coordinator, "parse_and_store_transcript"):
                # Convert speaker_info to list format expected by parser
                speaker_list = [speaker_info] if speaker_info else None
                await coordinator.parse_and_store_transcript(text, speaker_list)
        except Exception as e:
            # Log errors but don't let them affect transcription
            logger.warning(f"Transcript parsing update failed (non-critical): {e}")

    async def _get_end_buffer(self):
        """Get current end buffer value - to be implemented by coordinator."""
        return 0

    def finish_transcription(self):
        """Finish the transcription to get any remaining tokens."""
        # Flush any remaining accumulated text to parser
        if self.coordinator and self.accumulated_text_for_parser.strip():
            asyncio.create_task(self._update_coordinator_parser_async(self.coordinator, self.accumulated_text_for_parser, None))  # No specific speaker info for final flush
            self.accumulated_text_for_parser = ""
            logger.info(f"Flushed remaining accumulated text to parser")

        if self.online:
            try:
                return self.online.finish()
            except Exception as e:
                logger.warning(f"Failed to finish transcription: {e}")
        return None


class DiarizationProcessor:
    """Handles speaker diarization processing."""

    def __init__(self, args, diarization_obj):
        self.args = args
        self.diarization_obj = diarization_obj

    async def process(self, diarization_queue, get_state_callback, update_callback):
        """Process audio chunks for speaker diarization."""
        buffer_diarization = ""
        processed_chunks = 0

        logger.info("🔊 Diarization processor started")

        while True:
            try:
                pcm_array = await diarization_queue.get()
                if pcm_array is SENTINEL:
                    diarization_queue.task_done()
                    break

                processed_chunks += 1

                # Only log every 120 seconds to reduce spam significantly
                if not hasattr(self, "_last_diarization_log"):
                    self._last_diarization_log = 0
                current_time = time()
                if current_time - self._last_diarization_log > 120.0:
                    logger.info(f"🔊 Diarization: {processed_chunks} chunks processed")
                    self._last_diarization_log = current_time

                # Process diarization
                await self.diarization_obj.diarize(pcm_array)

                # Get current state and update speakers
                state = await get_state_callback()
                new_end = self.diarization_obj.assign_speakers_to_tokens(state["end_attributed_speaker"], state["tokens"])

                await update_callback(new_end, buffer_diarization)
                diarization_queue.task_done()

            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                if "pcm_array" in locals() and pcm_array is not SENTINEL:
                    diarization_queue.task_done()
        logger.info("Diarization processor finished.")

    def cleanup(self):
        """Clean up diarization resources."""
        if self.diarization_obj and hasattr(self.diarization_obj, "close"):
            self.diarization_obj.close()


class Formatter:
    """Handles formatting of transcription and diarization results."""

    def __init__(self, args):
        self.args = args

    async def format_by_sentences(self, tokens, sep, end_attributed_speaker, online=None):
        """Format tokens by sentence boundaries using the sentence tokenizer."""
        if not tokens:
            return []

        # Build full text from all tokens - optimize by filtering first
        token_texts = [token.text for token in tokens if token.text and token.text.strip()]
        if not token_texts:
            return []

        full_text = sep.join(token_texts)

        try:
            # Use the sentence tokenizer to split into sentences
            if online and hasattr(online, "tokenize") and online.tokenize:
                try:
                    # MosesSentenceSplitter expects a list input
                    sentence_texts = online.tokenize([full_text])
                except Exception as e:
                    # Fallback for other tokenizers that might expect string input
                    try:
                        sentence_texts = online.tokenize(full_text)
                    except Exception as e2:
                        logger.warning(f"Sentence tokenization failed: {e2}. Falling back to speaker-based segmentation.")
                        return await self.format_by_speaker(tokens, sep, end_attributed_speaker)
            else:
                # No tokenizer, split by basic punctuation
                sentence_texts = _sentence_split_regex.split(full_text)
                sentence_texts = [s.strip() for s in sentence_texts if s.strip()]

            if not sentence_texts:
                sentence_texts = [full_text]

            # Map sentences back to tokens and create lines
            lines = []
            token_index = 0

            for sent_text in sentence_texts:
                sent_text = sent_text.strip()
                if not sent_text:
                    continue

                # Find tokens that make up this sentence
                sent_tokens = []
                accumulated = ""
                start_token_index = token_index

                # Accumulate tokens until we roughly match the sentence text
                while token_index < len(tokens) and len(accumulated) < len(sent_text):
                    token = tokens[token_index]
                    if token.text.strip():  # Only consider non-empty tokens
                        accumulated = (accumulated + " " + token.text).strip() if accumulated else token.text
                        sent_tokens.append(token)
                    token_index += 1

                # If we didn't get any tokens, try to get at least one
                if not sent_tokens and start_token_index < len(tokens):
                    sent_tokens = [tokens[start_token_index]]
                    token_index = start_token_index + 1

                if sent_tokens:
                    # Determine speaker (use most common speaker in the sentence) - optimize speaker detection
                    if self.args.diarization:
                        # Filter valid speakers once
                        valid_speakers = [t.speaker for t in sent_tokens if t.speaker not in {-1} and t.speaker is not None]
                        if valid_speakers:
                            # Use most frequent speaker with optimized counting
                            speaker = max(set(valid_speakers), key=valid_speakers.count)
                        else:
                            speaker = sent_tokens[0].speaker
                    else:
                        speaker = 0  # Default speaker when no diarization (UI will show "Speaker 1")

                    # Create line for this sentence
                    line = {"speaker": speaker, "text": sent_text, "beg": format_time(sent_tokens[0].start), "end": format_time(sent_tokens[-1].end), "diff": 0}  # Not used in sentence mode
                    lines.append(line)

            return lines

        except Exception as e:
            logger.warning(f"Error in sentence-based formatting: {e}. Falling back to speaker-based segmentation.")
            return await self.format_by_speaker(tokens, sep, end_attributed_speaker)

    async def format_by_speaker(self, tokens, sep, end_attributed_speaker):
        """Format tokens by speaker changes (original behavior)."""
        previous_speaker = -1
        lines = []
        last_end_diarized = 0
        undiarized_text = []

        # Process each token
        for token in tokens:
            speaker = token.speaker

            # Handle diarization - optimize with set membership checks
            if self.args.diarization:
                speaker_in_invalid_set = speaker in {-1} or speaker is None
                if speaker_in_invalid_set and token.end >= end_attributed_speaker:
                    # Keep collecting undiarized text but also display it temporarily
                    undiarized_text.append(token.text)
                    # Assign a temporary positive speaker ID for display purposes
                    speaker = 0  # Use Speaker 0 as default for undiarized tokens (UI will show "Speaker 1")
                elif speaker_in_invalid_set and token.end < end_attributed_speaker:
                    speaker = previous_speaker if previous_speaker >= 0 else 0
                if not speaker_in_invalid_set:
                    last_end_diarized = max(token.end, last_end_diarized)

            # Group by speaker
            if speaker != previous_speaker or not lines:
                lines.append({"speaker": speaker, "text": token.text, "beg": format_time(token.start), "end": format_time(token.end), "diff": round(token.end - last_end_diarized, 2)})
                previous_speaker = speaker
            elif token.text:  # Only append if text isn't empty
                lines[-1]["text"] += sep + token.text
                lines[-1]["end"] = format_time(token.end)
                lines[-1]["diff"] = round(token.end - last_end_diarized, 2)

        return lines


class AudioProcessor:
    """
    Coordinates audio processing for transcription and diarization.
    Manages shared state and coordinates specialized processors.
    """

    def __init__(self):
        """Initialize the audio processor with configuration, models, and state."""

        models = AudioInsight()

        # Audio processing settings - cache computed values
        self.args = models.args

        # FFmpeg state - these were missing and are needed for the watchdog!
        self.sample_rate = 16000
        self.channels = 1
        self.bytes_per_sample = 2
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.sample_rate_str = str(self.sample_rate)  # Cache string conversion

        # Timing settings - needed for watchdog monitoring
        self.last_ffmpeg_activity = time()
        self.ffmpeg_health_check_interval = 10  # Less frequent health checks
        self.ffmpeg_max_idle_time = 30  # Allow much longer idle time before restart

        # State management
        self.is_stopping = False
        self.tokens = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""
        self.summaries = []  # Initialize summaries list
        self._has_summaries = False  # Efficient flag to track summary availability
        self._last_summary_check = 0  # Timestamp of last summary check

        # Initialize processor references as None - will be created on demand
        self.ffmpeg_processor = None
        self.transcription_processor = None
        self.diarization_processor = None
        self.formatter = Formatter(self.args)

        # Processing queues - will be created on demand
        self.transcription_queue = None
        self.diarization_queue = None

        # Task references
        self.transcription_task = None
        self.diarization_task = None
        self.ffmpeg_reader_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []

        # Initialize LLM inference processor early if enabled
        self.llm = None
        self.llm_task = None

        # Initialize transcript parser for structured processing
        self.transcript_parser = None
        self.parsed_transcripts = []  # Store parsed transcript data
        self.last_parsed_transcript = None  # Most recent parsed transcript
        self._parser_enabled = True  # Enable transcript parsing by default

        # Initialize LLM if enabled in features
        if getattr(self.args, "llm_inference", False):
            logger.info("🔧 Creating LLM processor on initialization")
            try:
                trigger_config = LLMTrigger(
                    summary_interval_seconds=getattr(self.args, "llm_summary_interval", 1.0),
                    new_text_trigger_chars=getattr(self.args, "llm_new_text_trigger", 100),
                )

                self.llm = LLMSummarizer(
                    model_id=getattr(self.args, "base_llm", "openai/gpt-4.1-mini"),
                    trigger_config=trigger_config,
                )
                self.llm.add_inference_callback(self._handle_inference_callback)
                logger.info(f"LLM inference initialized with model: {getattr(self.args, 'base_llm', 'openai/gpt-4.1-mini')}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM inference processor: {e}")
                self.llm = None

        # Initialize transcript parser
        try:
            parser_config = ParserConfig(model_id=getattr(self.args, "fast_llm", "openai/gpt-4.1-nano"), max_output_tokens=getattr(self.args, "parser_output_tokens", 33000), trigger_interval_seconds=getattr(self.args, "parser_trigger_interval", 1.0))
            self.transcript_parser = Parser(config=parser_config)
            logger.info(f"Transcript parser initialized with model: {getattr(self.args, 'fast_llm', 'openai/gpt-4.1-nano')}, max_output_tokens: {getattr(self.args, 'parser_output_tokens', 33000)}, trigger interval: {getattr(self.args, 'parser_trigger_interval', 1.0)}s")
        except Exception as e:
            logger.warning(f"Failed to initialize transcript parser: {e}")
            self.transcript_parser = None

        logger.info("🔧 AudioProcessor initialized - components will be created on demand")

    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        """Thread-safe update of transcription with new data."""
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep

    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        """Thread-safe update of diarization with new data."""
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization

    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.beg_loop
            self.tokens.append(ASRToken(start=current_time, end=current_time + 1, text=".", speaker=0, is_dummy=True))

    async def _handle_inference_callback(self, inference_response, transcription_text):
        """Handle callback when a inference result is generated."""
        logger.info(f"📝 Inference result generated: {inference_response.summary[:50]}...")
        logger.info(f"🔑 Key points: {', '.join(inference_response.key_points[:2])}...")

        # Store inference result in the response for client access
        async with self.lock:
            if not hasattr(self, "summaries"):
                self.summaries = []

            # Check for duplicate results (same result text)
            new_result = {
                "timestamp": time(),
                "summary": inference_response.summary,
                "key_points": inference_response.key_points,
                "text_length": len(transcription_text),
            }

            # Only add if it's not a duplicate
            is_duplicate = any(existing["summary"] == new_result["summary"] for existing in self.summaries)

            if not is_duplicate:
                self.summaries.append(new_result)
                self._has_summaries = True  # Set efficient flag
                logger.info(f"✅ Added new inference result (total: {len(self.summaries)})")
            else:
                logger.info("⚠️ Duplicate inference result detected, not adding")

    async def parse_and_store_transcript(self, text: str, speaker_info: Optional[list] = None, timestamps: Optional[dict] = None) -> Optional[ParsedTranscript]:
        """Parse transcript text and store it for sharing across the application.

        Args:
            text: Transcript text to parse
            speaker_info: Optional speaker information
            timestamps: Optional timestamp information

        Returns:
            ParsedTranscript: Parsed transcript data or None if parsing disabled/failed
        """
        if not self._parser_enabled or not self.transcript_parser or not text.strip():
            return None

        try:
            # Parse the transcript using the structured parser
            parsed_transcript = await self.transcript_parser.parse_transcript(text, speaker_info, timestamps)

            # Store the parsed transcript
            async with self.lock:
                self.parsed_transcripts.append(parsed_transcript)
                self.last_parsed_transcript = parsed_transcript

                # Keep only the last 50 parsed transcripts to prevent memory growth
                if len(self.parsed_transcripts) > 50:
                    self.parsed_transcripts = self.parsed_transcripts[-50:]

            # Update LLM with parsed text instead of raw text for better summaries
            if self.llm and parsed_transcript.parsed_text:
                self.llm.update_transcription(parsed_transcript.parsed_text, speaker_info[0] if speaker_info else None)

            logger.debug(f"📝 Parsed and stored transcript: {len(text)} -> {len(parsed_transcript.parsed_text)} chars")
            return parsed_transcript

        except Exception as e:
            logger.warning(f"Failed to parse transcript: {e}")
            return None

    def enable_transcript_parsing(self, enabled: bool = True):
        """Enable or disable transcript parsing.

        Args:
            enabled: Whether to enable transcript parsing
        """
        self._parser_enabled = enabled
        logger.info(f"Transcript parsing {'enabled' if enabled else 'disabled'}")

    def get_parsed_transcripts(self) -> list:
        """Get all parsed transcripts.

        Returns:
            List of ParsedTranscript objects
        """
        return self.parsed_transcripts.copy()

    def get_last_parsed_transcript(self) -> Optional[ParsedTranscript]:
        """Get the most recent parsed transcript.

        Returns:
            Most recent ParsedTranscript or None
        """
        return self.last_parsed_transcript

    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()

            # Calculate remaining times
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))

            # Only calculate remaining_diarization if diarization is enabled
            remaining_diarization = 0
            if self.args.diarization and self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 2))

            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_transcription,
                "remaining_time_diarization": remaining_diarization,
            }

    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            self.tokens.clear()
            self.buffer_transcription = self.buffer_diarization = ""
            self.end_buffer = self.end_attributed_speaker = 0
            self.full_transcription = self.last_response_content = ""
            self.beg_loop = time()
            # Reset parsed transcript data
            self.parsed_transcripts.clear()
            self.last_parsed_transcript = None

    async def results_formatter(self):
        """Format processing results for output."""
        while True:
            try:
                # Get current state
                state = await self.get_current_state()
                tokens = state["tokens"]
                buffer_transcription = state["buffer_transcription"]
                buffer_diarization = state["buffer_diarization"]
                end_attributed_speaker = state["end_attributed_speaker"]
                sep = state["sep"]

                # Add dummy tokens if needed
                if (not tokens or tokens[-1].is_dummy) and not self.args.transcription and self.args.diarization:
                    await self.add_dummy_token()
                    sleep(0.5)
                    state = await self.get_current_state()
                    tokens = state["tokens"]

                # Check if we have a sentence tokenizer available
                has_sentence_tokenizer = self.transcription_processor and hasattr(self.transcription_processor, "online") and self.transcription_processor.online and hasattr(self.transcription_processor.online, "tokenize") and self.transcription_processor.online.tokenize is not None

                # Format output - segment by sentences if tokenizer available, otherwise by speaker
                if has_sentence_tokenizer and tokens:
                    lines = await self.formatter.format_by_sentences(tokens, sep, end_attributed_speaker, self.transcription_processor.online)
                else:
                    lines = await self.formatter.format_by_speaker(tokens, sep, end_attributed_speaker)

                # Handle undiarized text
                undiarized_text = []
                if self.args.diarization:
                    # Collect any undiarized tokens
                    for token in tokens:
                        if (token.speaker in [-1] or token.speaker is None) and token.end >= end_attributed_speaker:
                            undiarized_text.append(token.text)

                if undiarized_text:
                    combined = sep.join(undiarized_text)
                    if buffer_transcription:
                        combined += sep
                    await self.update_diarization(end_attributed_speaker, combined)
                    buffer_diarization = combined

                # Create response object with s2hk conversion applied to final output
                if not lines:
                    lines = [{"speaker": 0, "text": "", "beg": format_time(0), "end": format_time(tokens[-1].end if tokens else 0), "diff": 0}]

                # Apply s2hk conversion to final output only (reduces latency) - optimize with list comprehension
                final_lines = [{**line, "text": s2hk(line["text"]) if line["text"] else line["text"]} for line in lines]

                final_buffer_transcription = s2hk(buffer_transcription) if buffer_transcription else buffer_transcription
                final_buffer_diarization = s2hk(buffer_diarization) if buffer_diarization else buffer_diarization

                response = {"lines": final_lines, "buffer_transcription": final_buffer_transcription, "buffer_diarization": final_buffer_diarization, "remaining_time_transcription": state["remaining_time_transcription"], "remaining_time_diarization": state["remaining_time_diarization"], "diarization_enabled": self.args.diarization}

                # Add summaries if available - optimize by checking flag first and limiting frequency
                current_time = time()
                if self._has_summaries or (current_time - self._last_summary_check > 5.0):  # OPTIMIZATION: Check every 5 seconds instead of 2
                    self._last_summary_check = current_time
                    async with self.lock:
                        if self.summaries:
                            response["summaries"] = self.summaries.copy()
                            # Only log occasionally to reduce spam
                            if current_time - getattr(self, "_last_summary_log", 0) > 60.0:  # Log every 60 seconds instead of 30
                                logger.info(f"Including {len(self.summaries)} summaries")
                                self._last_summary_log = current_time
                            self._has_summaries = True
                        else:
                            self._has_summaries = False

                # Add LLM inference processor stats if available
                if self.llm:
                    response["llm_stats"] = self.llm.get_stats()

                # Add parsed transcript data if available
                if self.transcript_parser and self._parser_enabled:
                    response["transcript_parser"] = {"enabled": True, "stats": self.transcript_parser.get_stats(), "last_parsed": self.last_parsed_transcript.model_dump() if self.last_parsed_transcript else None, "total_parsed": len(self.parsed_transcripts)}

                # Only yield if content has changed (use final converted content for comparison) - optimize string building
                response_content = " ".join(f"{line['speaker']} {line['text']}" for line in final_lines) + f" | {final_buffer_transcription} | {final_buffer_diarization}"

                if response_content != self.last_response_content and (final_lines or final_buffer_transcription or final_buffer_diarization):
                    yield response
                    self.last_response_content = response_content

                # Check for termination condition
                if self.is_stopping:
                    all_processors_done = True
                    if self.args.transcription and self.transcription_task and not self.transcription_task.done():
                        all_processors_done = False
                    if self.args.diarization and self.diarization_task and not self.diarization_task.done():
                        all_processors_done = False

                    if all_processors_done:
                        logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")

                        # Get final state and create final response
                        final_state = await self.get_current_state()

                        # Force commit any remaining buffer text for final processing
                        final_buffer_text = final_state["buffer_transcription"]
                        if final_buffer_text and self.transcription_processor:
                            # Try to finish the transcription to get any remaining tokens
                            remaining_transcript = self.transcription_processor.finish_transcription()
                            if remaining_transcript and remaining_transcript.text:
                                logger.info(f"Retrieved remaining transcript: '{remaining_transcript.text[:50]}...'")
                                # Update LLM with any remaining text
                                if self.llm:
                                    remaining_text_converted = s2hk(remaining_transcript.text) if remaining_transcript.text else ""
                                    if remaining_text_converted.strip():
                                        self.llm.update_transcription(remaining_text_converted, None)
                                        logger.info(f"Updated LLM with remaining text: '{remaining_text_converted[:50]}...'")

                        # Generate final inference if no summaries were created during processing
                        async with self.lock:
                            summaries_count = len(getattr(self, "summaries", []))

                        # Always generate a final comprehensive summary from the complete transcript
                        # This ensures we have a summary of the entire conversation, not just fragments
                        if self.llm and self.llm.accumulated_text.strip():
                            if summaries_count == 0:
                                logger.info("🔄 Generating final inference (no summaries created during processing)...")
                            else:
                                logger.info(f"🔄 Generating comprehensive final summary (had {summaries_count} intermediate summaries)...")

                            # Force a final summary of all accumulated text
                            await self.llm.force_inference()

                            # Wait for the final inference to complete and be processed by the callback
                            # Poll for up to 10 seconds to ensure the summary is added
                            max_wait_time = 10.0
                            poll_interval = 0.5
                            waited_time = 0.0
                            initial_summary_count = summaries_count

                            while waited_time < max_wait_time:
                                await asyncio.sleep(poll_interval)
                                waited_time += poll_interval

                                # Check if new summary was added
                                async with self.lock:
                                    current_summary_count = len(getattr(self, "summaries", []))

                                if current_summary_count > initial_summary_count:
                                    logger.info(f"✅ Final summary a~dded after {waited_time:.1f}s wait")
                                    break

                                if waited_time >= max_wait_time:
                                    logger.warning(f"⚠️ Final summary not added after {max_wait_time}s wait")
                                    break

                        # Get updated final state after inference processing
                        final_state = await self.get_current_state()

                        # Format final lines and apply refinement and s2hk conversion
                        final_lines_raw = await self.formatter.format_by_speaker(final_state["tokens"], final_state["sep"], final_state["end_attributed_speaker"])

                        # Apply s2hk conversion to output (refined or original)
                        final_lines_converted = [{**line, "text": s2hk(line["text"]) if line["text"] else line["text"]} for line in final_lines_raw]

                        # Refine and include any remaining buffer text in the final response
                        final_buffer_transcription = s2hk(final_state["buffer_transcription"]) if final_state["buffer_transcription"] else ""
                        final_buffer_diarization = s2hk(final_state["buffer_diarization"]) if final_state["buffer_diarization"] else ""

                        # Create final response with remaining buffer text included
                        final_response = {"lines": final_lines_converted, "buffer_transcription": final_buffer_transcription, "buffer_diarization": final_buffer_diarization, "remaining_time_transcription": 0, "remaining_time_diarization": 0, "diarization_enabled": self.args.diarization}

                        # Add existing summaries
                        async with self.lock:
                            if hasattr(self, "summaries") and self.summaries:
                                final_response["summaries"] = self.summaries.copy()
                                logger.info(f"🔄 Including {len(self.summaries)} final summaries in response")

                        # Add LLM stats
                        if self.llm:
                            final_response["llm_stats"] = self.llm.get_stats()

                        # Log final summary of transcription
                        total_lines = len(final_lines_converted)
                        buffer_chars = len(final_buffer_transcription) + len(final_buffer_diarization)
                        logger.info(f"📋 Final transcription: {total_lines} committed lines, {buffer_chars} buffer characters")

                        yield final_response
                        return

                await asyncio.sleep(0.2)  # OPTIMIZATION: Increased from 0.1 to reduce client update frequency

            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)  # Back off on error

    async def create_tasks(self):
        """Create async tasks for audio processing and result formatting."""
        self.all_tasks_for_cleanup = []  # Reset task list
        processing_tasks_for_watchdog = []

        # Initialize components on demand if not already created
        if not self.ffmpeg_processor:
            logger.info("🔧 Creating FFmpeg processor on demand")
            self.ffmpeg_processor = FFmpegProcessor(self.args)
            self.last_ffmpeg_activity = self.ffmpeg_processor.last_ffmpeg_activity

        # Initialize transcription components if needed - NOW WITH CORRECT LLM REFERENCE
        if self.args.transcription:
            if not self.transcription_processor:
                logger.info("🔧 Creating transcription processor on demand")
                models = AudioInsight()
                self.transcription_processor = TranscriptionProcessor(self.args, models.asr, models.tokenizer, coordinator=self)

            if not self.transcription_queue:
                self.transcription_queue = asyncio.Queue()

            # FIXED: Now self.llm is properly initialized before creating transcription task
            self.transcription_task = asyncio.create_task(self.transcription_processor.process(self.transcription_queue, self.update_transcription, self.llm))
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)

        # Initialize diarization components if needed
        if self.args.diarization:
            if not self.diarization_processor:
                logger.info("🔧 Creating diarization processor on demand")
                models = AudioInsight()
                self.diarization_processor = DiarizationProcessor(self.args, models.diarization)

            if not self.diarization_queue:
                self.diarization_queue = asyncio.Queue()

            self.diarization_task = asyncio.create_task(self.diarization_processor.process(self.diarization_queue, self.get_current_state, self.update_diarization))
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)

        self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_processor.read_audio_data(self.transcription_queue, self.diarization_queue))
        self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
        processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        # Start LLM inference processor monitoring if enabled - AFTER initialization
        if self.llm:
            self.llm_task = asyncio.create_task(self.llm.start_monitoring())
            self.all_tasks_for_cleanup.append(self.llm_task)
            logger.info("LLM inference processor monitoring task started")

        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)

        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        while True:
            try:
                await asyncio.sleep(15)  # Check every 15 seconds instead of 10
                current_time = time()

                for i, task in enumerate(tasks_to_monitor):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, "get_name") else f"Task {i}"
                        if exc:
                            logger.error(f"{task_name} failed: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally")

                # Sync FFmpeg activity timing
                self.last_ffmpeg_activity = self.ffmpeg_processor.last_ffmpeg_activity
                ffmpeg_idle_time = current_time - self.last_ffmpeg_activity

                # Only log idle warnings every 60 seconds and if significant
                if not hasattr(self, "_last_idle_warning"):
                    self._last_idle_warning = 0

                if ffmpeg_idle_time > 20 and current_time - self._last_idle_warning > 60.0:
                    logger.warning(f"FFmpeg idle for {ffmpeg_idle_time:.1f}s")
                    self._last_idle_warning = current_time

                if ffmpeg_idle_time > 30 and not self.is_stopping:
                    logger.error("FFmpeg idle too long, forcing restart")
                    await self.ffmpeg_processor.restart_ffmpeg()
                    # Update timing after restart
                    self.last_ffmpeg_activity = self.ffmpeg_processor.last_ffmpeg_activity
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")

        # Stop LLM inference processor first to generate final inference
        if self.llm:
            await self.llm.stop_monitoring()
            logger.info("LLM inference processor stopped")

        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()

        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")

        # Clean up specialized processors
        if self.ffmpeg_processor:
            await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_processor.cleanup)

        if self.diarization_processor:
            self.diarization_processor.cleanup()

        logger.info("AudioProcessor cleanup complete.")

    async def force_reset(self):
        """Force reset all resources and clear memory for a fresh session.

        This is more aggressive than cleanup() and ensures no memory leaks between sessions.
        """
        logger.info("🧹 Starting force reset of AudioProcessor...")

        # Stop everything aggressively
        self.is_stopping = True

        # Cancel all tasks immediately
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()

        # Wait for tasks to complete cancellation
        if self.all_tasks_for_cleanup:
            try:
                await asyncio.gather(*[t for t in self.all_tasks_for_cleanup if t], return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error waiting for task cancellation: {e}")

        # Clear all queues aggressively
        if self.transcription_queue:
            while not self.transcription_queue.empty():
                try:
                    self.transcription_queue.get_nowait()
                    self.transcription_queue.task_done()
                except:
                    break

        if self.diarization_queue:
            while not self.diarization_queue.empty():
                try:
                    self.diarization_queue.get_nowait()
                    self.diarization_queue.task_done()
                except:
                    break

        # Force cleanup all processors with proper cleanup
        if self.ffmpeg_processor:
            try:
                # Ensure FFmpeg cleanup happens synchronously and completely
                await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_processor.cleanup)
                # Wait a moment for cleanup to complete
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error cleaning up FFmpeg processor: {e}")
            self.ffmpeg_processor = None

        if self.transcription_processor:
            self.transcription_processor = None

        if self.diarization_processor:
            try:
                self.diarization_processor.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up diarization processor: {e}")
            self.diarization_processor = None

        if self.llm:
            try:
                await self.llm.stop_monitoring()
            except Exception as e:
                logger.warning(f"Error stopping LLM monitoring: {e}")
            self.llm = None

        # Clear all memory buffers and state
        async with self.lock:
            self.tokens.clear()
            self.buffer_transcription = ""
            self.buffer_diarization = ""
            self.full_transcription = ""
            self.end_buffer = 0
            self.end_attributed_speaker = 0
            self.last_response_content = ""
            if hasattr(self, "summaries"):
                self.summaries.clear()
            self._has_summaries = False
            # Clear parsed transcript data
            if hasattr(self, "parsed_transcripts"):
                self.parsed_transcripts.clear()
            self.last_parsed_transcript = None

        # Clear task references
        self.transcription_task = None
        self.diarization_task = None
        self.ffmpeg_reader_task = None
        self.watchdog_task = None
        self.llm_task = None
        self.all_tasks_for_cleanup.clear()

        # Reset queues to None - will be recreated on demand
        self.transcription_queue = None
        self.diarization_queue = None

        # Reset timing and state flags
        self.beg_loop = time()
        self.is_stopping = False

        # DO NOT re-initialize components here - let them be created on demand
        # This avoids the FFmpeg restart loop issue
        logger.info("🧹 Force reset completed - components will be re-initialized on demand")

    async def process_audio(self, message):
        """Process incoming audio data."""
        # If already stopping or stdin is closed, ignore further audio, especially residual chunks.
        if self.is_stopping or (self.ffmpeg_processor.ffmpeg_process and self.ffmpeg_processor.ffmpeg_process.stdin and self.ffmpeg_processor.ffmpeg_process.stdin.closed):
            logger.warning(f"AudioProcessor is stopping or stdin is closed. Ignoring incoming audio message (length: {len(message)}).")
            if not message and self.ffmpeg_processor.ffmpeg_process and self.ffmpeg_processor.ffmpeg_process.stdin and not self.ffmpeg_processor.ffmpeg_process.stdin.closed:
                logger.info("Received empty message while already in stopping state; ensuring stdin is closed.")
                try:
                    self.ffmpeg_processor.ffmpeg_process.stdin.close()
                except Exception as e:
                    logger.warning(f"Error closing ffmpeg stdin on redundant stop signal during stopping state: {e}")
            return

        if not message:  # primary signal to start stopping
            logger.info("Empty audio message received, initiating stop sequence.")
            self.is_stopping = True
            if self.ffmpeg_processor.ffmpeg_process and self.ffmpeg_processor.ffmpeg_process.stdin and not self.ffmpeg_processor.ffmpeg_process.stdin.closed:
                try:
                    self.ffmpeg_processor.ffmpeg_process.stdin.close()
                    logger.info("FFmpeg stdin closed due to primary stop signal.")
                except Exception as e:
                    logger.warning(f"Error closing ffmpeg stdin on stop: {e}")
            return

        await self.ffmpeg_processor.process_audio_chunk(message)
