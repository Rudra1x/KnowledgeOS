# loaders/youtube_loader.py

import re
import uuid
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from core import Document, Loader


class YouTubeLoader(Loader):
    """
    Loads a YouTube video's transcript.

    Emits ONE Document per transcript segment (default), OR one merged Document
    covering the whole video (mode='full'). Segment mode preserves timestamps
    for precise citations ('at 3:42'); full mode is simpler when timestamps
    don't matter.

    URL formats accepted:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    - Just the raw video ID

    Handles gracefully:
    - Videos with no transcripts (TranscriptsDisabled)
    - Videos with no transcript in requested languages (NoTranscriptFound)
    - Unavailable / private / deleted videos (VideoUnavailable)
    """

    def __init__(
        self,
        mode:      str        = "segment",     # 'segment' or 'full'
        languages: list[str]  = ["en", "en-US", "en-GB"],
        min_chars: int        = 5,
    ):
        if mode not in {"segment", "full"}:
            raise ValueError(f"mode must be 'segment' or 'full', got {mode!r}")
        self.mode      = mode
        self.languages = languages
        self.min_chars = min_chars

    def load(self, source: str) -> list[Document]:
        video_id = self._extract_video_id(source)

        try:
            api        = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=self.languages)
        except TranscriptsDisabled:
            print(f"[YouTubeLoader] Transcripts disabled for video {video_id}")
            return []
        except NoTranscriptFound:
            print(f"[YouTubeLoader] No transcript found in languages {self.languages} for {video_id}")
            return []
        except VideoUnavailable:
            print(f"[YouTubeLoader] Video unavailable: {video_id}")
            return []

        segments = transcript.to_raw_data()   # list of {'text', 'start', 'duration'}

        if self.mode == "full":
            return self._as_full(video_id, source, segments)
        return self._as_segments(video_id, source, segments)

    # ------------------------------------------------------------------
    def _as_segments(self, video_id: str, source: str, segments: list[dict]) -> list[Document]:
        docs = []
        for seg in segments:
            text = seg["text"].strip()
            if len(text) < self.min_chars:
                continue

            start = float(seg["start"])
            docs.append(Document(
                doc_id   = str(uuid.uuid4()),
                content  = text,
                source   = source,
                metadata = {
                    "file_type":          "youtube",
                    "content_type":       "transcript_segment",
                    "video_id":           video_id,
                    "start_time_seconds": start,
                    "duration_seconds":   float(seg.get("duration", 0.0)),
                    "timestamp":          self._fmt_timestamp(start),
                    "video_url":          f"https://youtu.be/{video_id}?t={int(start)}",
                },
            ))
        return docs

    def _as_full(self, video_id: str, source: str, segments: list[dict]) -> list[Document]:
        text = " ".join(seg["text"].strip() for seg in segments if seg.get("text"))
        text = text.strip()
        if len(text) < self.min_chars:
            return []

        duration = float(segments[-1]["start"] + segments[-1].get("duration", 0.0)) if segments else 0.0
        return [Document(
            doc_id   = str(uuid.uuid4()),
            content  = text,
            source   = source,
            metadata = {
                "file_type":         "youtube",
                "content_type":      "transcript_full",
                "video_id":          video_id,
                "duration_seconds":  duration,
                "segment_count":     len(segments),
                "video_url":         f"https://youtu.be/{video_id}",
            },
        )]

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_video_id(source: str) -> str:
        """Handle all common YouTube URL shapes, or a raw 11-char ID."""
        # Raw video ID
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", source):
            return source

        parsed = urlparse(source)

        # youtu.be/VIDEO_ID
        if parsed.netloc == "youtu.be":
            return parsed.path.lstrip("/")

        # youtube.com/watch?v=VIDEO_ID
        if "youtube.com" in parsed.netloc:
            if parsed.path == "/watch":
                qs = parse_qs(parsed.query)
                if "v" in qs:
                    return qs["v"][0]
            # /shorts/VIDEO_ID or /embed/VIDEO_ID
            m = re.match(r"/(shorts|embed)/([A-Za-z0-9_-]{11})", parsed.path)
            if m:
                return m.group(2)

        raise ValueError(f"Could not extract YouTube video ID from: {source}")

    @staticmethod
    def _fmt_timestamp(seconds: float) -> str:
        """Format seconds as H:MM:SS or M:SS."""
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"