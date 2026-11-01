#!/usr/bin/env python3
"""
ZentroMart VoiceOps 2026 — Official Batch Audio CLI Runner
Usage:
    python run.py <input_audio_folder> <output_folder>
    python run.py --input_dir <input_audio_folder> --output_dir <output_folder>
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient
from src.agent import ZentroMartVoiceAgent, save_json

def detect_tts_language(text, stt_lang="en-IN"):
    if re.search(r'[\u0900-\u097F]', text):
        return "hi-IN"
    u_words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    if u_words & {'hai', 'hain', 'kya', 'mein', 'aapka', 'kripya', 'nahi', 'kar', 'sakte'}:
        return "hi-en"
    return stt_lang if stt_lang in ("en-IN", "hi-IN") else "en-IN"

def voice_turn(audio_path, agent, stt, tts, stt_language="en-IN", output_file="reply.wav", voice="Kaveri", model="timbre-v2.5"):
    stt_res = stt.transcribe(str(audio_path), language_code=stt_language)
    heard = stt_res.get("transcript", "") if isinstance(stt_res, dict) else str(stt_res)

    if not heard.strip():
        reply = "Sorry, I didn't catch that. Could you say it again?"
        spoken_reply = reply
        agent.last_turn_metadata = {
            'turn': agent.state['turn'] + 1,
            'heard': '',
            'text_reply': reply,
            'spoken_reply': spoken_reply,
            'retrieved_sources': [],
            'retrieved_sections': [],
            'tool_calls': []
        }
    else:
        reply = agent.respond(heard)
        spoken_reply = agent.last_turn_metadata['spoken_reply']

    tts_lang = detect_tts_language(spoken_reply, stt_language)
    try:
        tts.synthesize(spoken_reply, voice=voice, model=model, language=tts_lang, output_file=str(output_file))
    except Exception:
        tts.synthesize(spoken_reply, voice=voice, model=model, language="en-IN", output_file=str(output_file))

    return heard, reply

def run_batch_audio_folder(input_dir, output_dir="sample_outputs", stt_language="en-IN", voice="Kaveri", model="timbre-v2.5", multi_turn_by_prefix=True):
    if not os.environ.get("GNANI_API_KEY"):
        raise RuntimeError("GNANI_API_KEY environment variable is not set. Export it before running run.py.")

    in_path, out_path = Path(input_dir), Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    audio_files = sorted([
        p for p in in_path.iterdir()
        if p.suffix.lower() in ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    ])
    if not audio_files:
        print(f"No audio files found in {in_path}")
        return []

    stt = GnaniSTTClient()
    tts = GnaniTTSClient()
    agent = ZentroMartVoiceAgent()
    batch_logs, current_conv_prefix = [], None

    for audio_file in audio_files:
        stem = audio_file.stem
        if multi_turn_by_prefix and '_' in stem:
            prefix = stem.split('_')[0]
            if prefix != current_conv_prefix:
                agent.reset()
                current_conv_prefix = prefix
        else:
            agent.reset()

        reply_wav_path = out_path / f"{stem}_reply.wav"
        reply_txt_path = out_path / f"{stem}_reply.txt"

        heard, reply = voice_turn(
            audio_path=audio_file, agent=agent, stt=stt, tts=tts,
            stt_language=stt_language, output_file=reply_wav_path,
            voice=voice, model=model
        )

        meta = agent.last_turn_metadata
        reply_txt_path.write_text(reply, encoding='utf-8')

        record = {
            'input_audio': audio_file.name,
            'output_audio': reply_wav_path.name,
            'output_text_file': reply_txt_path.name,
            'transcript': heard,
            'reply_text': reply,
            'spoken_text': meta['spoken_reply'],
            'retrieved_sources': meta['retrieved_sources'],
            'retrieved_sections': meta['retrieved_sections'],
            'tool_calls': [t.get('call_str', t.get('tool')) for t in meta['tool_calls']]
        }
        batch_logs.append(record)
        print(f"Processed {audio_file.name} -> {reply_wav_path.name} | Chunks: {record['retrieved_sections']}")

    save_json(out_path / "batch_results.json", batch_logs)
    print(f"\nSaved {len(batch_logs)} audio/text replies and chunk logs to {out_path}")
    return batch_logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ZentroMart VoiceOps Agent on a folder of customer audio files.")
    parser.add_argument("input_pos", nargs="?", help="Input folder containing customer audio files")
    parser.add_argument("output_pos", nargs="?", help="Output folder to save text/audio replies and retrieved chunks")
    parser.add_argument("--input_dir", "-i", default=None, help="Input folder containing customer audio files")
    parser.add_argument("--output_dir", "-o", default="sample_outputs", help="Output folder for replies")
    parser.add_argument("--lang", default="en-IN", help="Gnani STT language code (default: en-IN)")
    parser.add_argument("--voice", default="Kaveri", help="Gnani TTS voice name (default: Kaveri)")
    args = parser.parse_args()

    in_folder = args.input_dir or args.input_pos
    out_folder = args.output_pos or args.output_dir
    if not in_folder:
        parser.error("Please provide an input audio folder path (e.g., `python run.py ./input_audio ./sample_outputs`).")

    run_batch_audio_folder(in_folder, out_folder, stt_language=args.lang, voice=args.voice)
