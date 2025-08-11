# Copyright (c) 2024–2025, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Pipecat Twilio Phone Example with OpenAI Function Calling + ElevenLabs TTS.

Run:
    python bot.py -t twilio -x your-public-hostname.example.com
"""

import os
import httpx
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from deepgram import LiveOptions
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# Function calling schemas
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

load_dotenv(override=True)


async def run_bot(transport: BaseTransport):
    logger.info("Starting bot")

    # --- Providers ---
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        sample_rate=int(os.getenv("AUDIO_IN_SAMPLE_RATE", "8000")),
        live_options=LiveOptions(
            language="multi",
            model=os.getenv("DEEPGRAM_MODEL", "nova-3-general"),
            interim_results=True,
            smart_format=True,
            punctuate=True,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),  # "Rachel" default
        model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
        sample_rate=int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "8000")),
    )

    llm = OLLamaLLMService(model=os.getenv("OLLAMA_MODEL","gpt-oss:20b"), base_url=os.getenv("OLLAMA_BASE_URL","http://localhost:11434/v1"))

    # --- Function calling: tool schemas ---
    visiting_hours_fn = FunctionSchema(
        name="get_visiting_hours",
        description="Return visiting hours for a hospital department",
        properties={
            "department": {
                "type": "string",
                "description": "Hospital department name (e.g., 'ICU', 'pediatrics', 'oncology').",
            }
        },
        required=[],
    )

    availability_fn = FunctionSchema(
        name="check_appointment_availability",
        description="Fetch appointment availability for MRI in Radiology for a given date (YYYY-MM-DD).",
        properties={
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD",
            }
        },
        required=["date"],
    )

    tools = ToolsSchema(standard_tools=[visiting_hours_fn, availability_fn])

    # --- Tool: get_visiting_hours (mock table) ---
    async def get_visiting_hours(params: FunctionCallParams):
        dept = (params.arguments or {}).get("department", "")
        d = (dept or "").strip().lower()
        table = {
            "icu":        {"department": "ICU",        "weekday": "11:00–19:00", "weekend": "12:00–18:00", "notes": "Max 2 visitors; 15 min per visit."},
            "pediatrics": {"department": "Pediatrics", "weekday": "10:00–20:00", "weekend": "10:00–18:00", "notes": "Parents/guardians allowed anytime."},
            "oncology":   {"department": "Oncology",   "weekday": "10:00–19:00", "weekend": "11:00–17:00", "notes": "Masks required."},
        }
        result = table.get(d, {"department": dept or "General", "weekday": "09:00–20:00", "weekend": "10:00–18:00", "notes": "Check with the nurse station for exceptions."})
        logger.info("TOOL CALL: get_visiting_hours args=%r -> %r", params.arguments, result)
        await params.result_callback(result)

    # --- Tool: check_appointment_availability (HTTP GET) ---
    API_URL = "https://kikrankenhausms-production.up.railway.app/api/availability"

    async def check_appointment_availability(params: FunctionCallParams):
        date = (params.arguments or {}).get("date", "").strip()
        if not date:
            await params.result_callback({"error": "missing_date", "message": "Please provide a date in YYYY-MM-DD."})
            return
        query = {"date": date, "appointment_type": "MRI", "department": "Radiology"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(API_URL, params=query)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            logger.warning("availability request failed: %s", e)
            await params.result_callback({"error": "request_failed", "details": str(e)})
            return
        result = {"date": date, "appointment_type": "MRI", "department": "Radiology", "availability": data}
        logger.info("TOOL CALL: check_appointment_availability args=%r -> OK", params.arguments)
        await params.result_callback(result)

    # Register tools
    if hasattr(llm, "register_function"):
        llm.register_function("get_visiting_hours", get_visiting_hours)
        llm.register_function("check_appointment_availability", check_appointment_availability)

    # --- Prompt / persona (nudges tool use) ---
    messages = [
        {
            "role": "system",
            "content": (
                "You are the warm, professional receptionist for Starlight General Hospital. "
                "If a caller asks about visiting hours, call the get_visiting_hours tool with the "
                "department name, then summarize briefly for voice. "
                "If a caller asks about MRI appointment availability in Radiology, ask for a date "
                "(YYYY-MM-DD) if missing, then call the check_appointment_availability tool and summarize the result. "
                "Do not give medical advice; offer to connect callers to the right department."
            ),
        },
    ]

    context = OpenAILLMContext(messages, tools=tools)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=int(os.getenv("AUDIO_IN_SAMPLE_RATE", "8000")),
            audio_out_sample_rate=int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "8000")),
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append({"role": "system", "content": "Greet the caller, introduce yourself, and ask how you can help."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
