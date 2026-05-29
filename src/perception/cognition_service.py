from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Protocol

from src.perception import CognitionEngine, CognitionError, CognitionResult, DEFAULT_MODEL_ID
from src.safety.command_validator import CommandValidator, DroneOperationalState, DroneStateSnapshot


logger = logging.getLogger(__name__)

_WORKER_ENGINE: Optional[CognitionEngine] = None


class AsyncCognitionService(Protocol):
    async def warmup(self) -> None:
        ...

    async def process_intent(
        self,
        text_input: str,
        image_path: Optional[str] = None,
        *,
        drone_state: Optional[DroneStateSnapshot] = None,
    ) -> CognitionResult | CognitionError:
        ...

    async def close(self) -> None:
        ...


class ThreadedCognitionService:
    def __init__(
        self,
        engine: CognitionEngine,
        *,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.engine = engine
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="edge-vla-cognition",
        )
        self._owns_executor = executor is None

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.engine.warmup)

    async def process_intent(
        self,
        text_input: str,
        image_path: Optional[str] = None,
        *,
        drone_state: Optional[DroneStateSnapshot] = None,
    ) -> CognitionResult | CognitionError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(
                self.engine.process_intent,
                text_input,
                image_path,
                drone_state=drone_state,
            ),
        )

    async def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=False)


class ProcessCognitionService:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        max_tokens: int = 128,
        temperature: float = 0.0,
        log_path: str | Path = "logs/cognition.log",
        timeout_s: float = 75.0,
        vlm_backend: str = "mlx",
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.log_path = str(log_path)
        self.timeout_s = timeout_s
        self.vlm_backend = vlm_backend
        self._executor: Optional[ProcessPoolExecutor] = None
        self._prompt_engine = CognitionEngine(
            validator=_default_validator(),
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            log_path=log_path,
            vlm_backend=vlm_backend,
        )

    async def warmup(self) -> None:
        try:
            await self._run_worker(_worker_warmup)
        except Exception as exc:  # noqa: BLE001 - native worker failures must not kill the agent.
            self._reset_executor()
            raise RuntimeError(f"cognition worker warmup failed: {exc}") from exc

    async def process_intent(
        self,
        text_input: str,
        image_path: Optional[str] = None,
        *,
        drone_state: Optional[DroneStateSnapshot] = None,
    ) -> CognitionResult | CognitionError:
        try:
            drone_state_payload = drone_state.model_dump(mode="json") if drone_state is not None else None
            return await self._run_worker(
                _worker_process_intent,
                text_input,
                image_path,
                drone_state_payload,
            )
        except asyncio.TimeoutError:
            self._reset_executor()
            details = f"cognition worker timed out after {self.timeout_s:.1f}s"
            logger.error(details)
        except BrokenProcessPool as exc:
            self._reset_executor()
            details = f"cognition worker crashed or was terminated: {exc}"
            logger.error(details)
        except Exception as exc:  # noqa: BLE001
            self._reset_executor()
            details = f"cognition worker failed: {exc}"
            logger.exception("Cognition worker failed")

        return CognitionError(
            reason="INFERENCE_ERROR",
            raw_prompt=self._fallback_prompt(text_input, image_path, drone_state),
            raw_response="",
            parsed_json=None,
            details=details,
        )

    async def close(self) -> None:
        self._reset_executor(wait=True)

    async def _run_worker(self, fn, *args):
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._ensure_executor(), partial(fn, *args))
        return await asyncio.wait_for(future, timeout=self.timeout_s)

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            context = get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=_init_worker,
                initargs=(self.model_id, self.max_tokens, self.temperature, self.log_path, self.vlm_backend),
            )
        return self._executor

    def _reset_executor(self, *, wait: bool = False) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def _fallback_prompt(
        self,
        text_input: str,
        image_path: Optional[str],
        drone_state: Optional[DroneStateSnapshot],
    ) -> str:
        return self._prompt_engine._build_prompt(  # noqa: SLF001 - avoids losing blackbox context after worker aborts.
            text_input,
            image_path=image_path,
            drone_state=drone_state,
        )


def _default_validator() -> CommandValidator:
    return CommandValidator(
        DroneStateSnapshot(
            state=DroneOperationalState.GROUNDED,
            connected=False,
            battery_remaining=None,
        )
    )


def _init_worker(model_id: str, max_tokens: int, temperature: float, log_path: str, vlm_backend: str) -> None:
    global _WORKER_ENGINE  # noqa: PLW0603
    _WORKER_ENGINE = CognitionEngine(
        validator=_default_validator(),
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        log_path=log_path,
        vlm_backend=vlm_backend,
    )


def _worker_engine() -> CognitionEngine:
    if _WORKER_ENGINE is None:
        _init_worker(DEFAULT_MODEL_ID, 128, 0.0, "logs/cognition.log", "mlx")
    assert _WORKER_ENGINE is not None
    return _WORKER_ENGINE


def _worker_warmup() -> None:
    _worker_engine().warmup()


def _worker_process_intent(
    text_input: str,
    image_path: Optional[str],
    drone_state_payload: Optional[dict],
) -> CognitionResult | CognitionError:
    drone_state = (
        DroneStateSnapshot.model_validate(drone_state_payload)
        if drone_state_payload is not None
        else None
    )
    return _worker_engine().process_intent(
        text_input,
        image_path=image_path,
        drone_state=drone_state,
    )
