"""Session/request route selection for the API-server adapter."""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("gateway.platforms.api_server")


class SessionRuntimeMixin:
    """Keep explicit session choices separate from the last executed model."""

    def _stored_session_route(self, session: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(session, dict):
            return None
        stored = session.get("model")
        config = self._parse_session_model_config(session.get("model_config"))
        lock = config.get("browser_model_lock")
        if isinstance(lock, dict):
            if lock.get("confirmed"):
                return None
            stored, config = lock.get("model"), lock
            if lock.get("route_source") == "model_routes":
                route = self._resolve_route(stored)
                if not route:
                    return None
                # Keep credentials in configured routes, never in durable session metadata.
                return {**route, **{k: lock[k] for k in ("provider", "base_url", "api_mode") if lock.get(k)}}
            # Unconfirmed model-only locks cannot identify their historical provider.
            if not lock.get("provider"):
                return None
        else:
            # update_session_model writes an explicit route in both CLI/TUI shapes.
            # Provider recovery idea from NousResearch/hermes-agent PR #123802.
            nested = config.get("gateway_runtime")
            if not config.get("provider") and isinstance(nested, dict):
                config = nested
        if stored and stored != self._model_name:
            if config.get("provider"):
                return {"model": stored, **{k: config[k] for k in ("provider", "base_url", "api_mode")
                                            if config.get(k)}}
            return self._resolve_route(stored)
        return None

    @staticmethod
    def _provider_selection_model(
        provider: Optional[str], default_model: str, default_provider: Optional[str],
        runtime: Optional[Dict[str, Any]] = None,
    ) -> str:
        if provider == default_provider:
            return default_model
        from hermes_cli.models import get_default_model_for_provider
        return (runtime or {}).get("model") or get_default_model_for_provider(provider) or ""

    def _complete_provider_only_request(self, runtime_request: Dict[str, Any]) -> None:
        requested = runtime_request.get("requested") or {}
        if requested.get("provider") and not requested.get("model"):
            lock = self._complete_session_selection(runtime_request)
            runtime_request["route"] = (
                {k: lock[k] for k in ("model", "provider", "base_url", "api_mode") if lock.get(k)}
                if lock else None)
            runtime_request["route_source"] = "raw_request"

    def _complete_session_selection(self, runtime_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Capture route identity at selection time; credentials never enter the lock."""
        from gateway.run import _load_gateway_config, _resolve_gateway_model, _resolve_runtime_agent_kwargs

        requested = runtime_request.get("requested") or {}
        route = runtime_request.get("route") or {}
        try:
            default_runtime = _resolve_runtime_agent_kwargs()
        except RuntimeError:
            config = _load_gateway_config().get("model") or {}
            default_runtime = config if isinstance(config, dict) else {}
        default_provider = default_runtime.get("provider") or default_runtime.get("requested_provider")
        provider = requested.get("provider") or route.get("provider") or default_provider
        model = requested.get("model") or route.get("model")
        if not model:
            model = self._provider_selection_model(
                provider, default_runtime.get("model") or _resolve_gateway_model(), default_provider)
        runtime = (self._resolve_provider_runtime(provider, target_model=route.get("model") or model, required=False)
                   or {}) if provider else default_runtime
        if provider == "auto":
            provider = runtime.get("provider")
            if provider == "auto":
                provider = None
        if not model:
            model = self._provider_selection_model(
                provider, _resolve_gateway_model(), default_provider, runtime)
            if not model:
                return None
        lock = {
            "model": model or "", "provider": provider or "",
            "model_options": runtime_request.get("model_options") or {},
            "route_source": runtime_request.get("route_source") or "",
            "confirmed": bool(runtime_request.get("require_model_lock")), "updated_at": time.time()}
        for key in ("base_url", "api_mode"):
            value = route.get(key) or runtime.get(key)
            if value:
                lock[key] = value
        return lock

    def _persist_initial_session_selection(
        self, session: Dict[str, Any], runtime_request: Dict[str, Any], session_key: Optional[str] = None,
    ) -> None:
        requested = runtime_request.get("requested") or {}
        config = self._parse_session_model_config(session.get("model_config"))
        if not (requested.get("model") or requested.get("provider")) or config.get("browser_model_lock"):
            return
        if self._stored_session_route(session):
            return
        if self._session_model_override_for(session_key or session["id"]):
            return
        lock = self._complete_session_selection(runtime_request)
        if lock:
            self._ensure_session_db().patch_session_model_config(session["id"], {"browser_model_lock": lock})

    def _session_model_override_for(self, session_key: Optional[str]) -> Optional[Dict[str, Any]]:
        """The gateway's per-session ``/model`` override for *session_key*, if any — a
        user-issued ``/model`` always wins over static route config."""
        if not session_key:
            return None
        try:
            from gateway.run import _gateway_runner_ref
            runner = _gateway_runner_ref()
            if runner is None:
                return None
            store = getattr(runner, "session_store", None)
            if store is not None and store.lookup_by_session_key(session_key) is None:
                from hermes_constants import get_hermes_home
                from gateway.config import Platform
                entry = store.lookup_by_session_id(session_key, owner_home=get_hermes_home(), platform=Platform.API_SERVER)
                if entry is not None:
                    session_key = entry.session_key
            try:
                rehydrate = getattr(runner, "_rehydrate_session_model_override", None)
                if callable(rehydrate):
                    rehydrate(session_key)
            except Exception:
                logger.debug(
                    "api_server failed to rehydrate session /model override for %s", session_key, exc_info=True)
            override = runner._session_model_overrides.get(session_key)
            return dict(override) if isinstance(override, dict) else None
        except Exception:
            return None

    def _select_agent_runtime(
        self, runtime_kwargs: Dict[str, Any], model: str, *, requested_model: Optional[str],
        requested_provider: Optional[str], route: Optional[Dict[str, Any]], session_model: Optional[str],
        confirmed_runtime_lock: bool, gateway_session_key: Optional[str], session_id: Optional[str]) -> tuple:
        """Apply the model/provider precedence chain for one agent (mutates ``runtime_kwargs``):
        confirmed Browser lock > session ``/model`` override > explicit API session selection >
        model_routes alias > per-request provider/model > global defaults. A confirmed lock
        bypasses the override and fails closed if its provider cannot be resolved.
        Returns ``(model, session_override, request_model, request_provider)``."""
        from gateway.platforms.api_server import _apply_runtime_agent_overrides, _clean_request_string

        request_model = _clean_request_string(requested_model)
        request_provider = _clean_request_string(requested_provider)
        route_cfg = route if isinstance(route, dict) else {}
        route_model = _clean_request_string(route_cfg.get("model"))
        route_provider = _clean_request_string(route_cfg.get("provider"))
        session_key = gateway_session_key or session_id
        session_row_model = _clean_request_string(session_model)
        current_provider = _clean_request_string(runtime_kwargs.get("provider"))
        session_override = None if confirmed_runtime_lock else self._session_model_override_for(session_key)
        # Model-string precedence (override > session-persisted > global) is owned by
        # hermes_cli.model_switch.resolve_effective_model.
        from hermes_cli.model_switch import resolve_effective_model
        if session_override:
            override_model = resolve_effective_model(session_override, None, model)
            # A rehydrated override without credentials must resolve on its OWN provider.
            # If unavailable, retain the complete default route and retry the durable pin next turn.
            if session_override.get("api_key"):
                from gateway.run import _credential_pool_for_provider
                provider_runtime = dict(session_override)
                if provider_runtime.get("credential_pool") is None:
                    provider_runtime["credential_pool"] = _credential_pool_for_provider(session_override.get("provider"))
            else:
                provider_runtime = self._resolve_provider_runtime(
                    _clean_request_string(session_override.get("provider")) or current_provider,
                    target_model=override_model, required=False)
            if provider_runtime:
                model = override_model
                runtime_kwargs.clear()
                runtime_kwargs.update({k: provider_runtime[k] for k in (
                    "provider", "requested_provider", "api_key", "base_url", "api_mode",
                    "max_tokens", "credential_pool", "request_overrides", "capabilities",
                ) if k in provider_runtime})
                _apply_runtime_agent_overrides(runtime_kwargs, session_override)
            else:
                logger.warning(
                    "Session /model provider %s unavailable; using default model=%s provider=%s",
                    session_override.get("provider"), model, current_provider)
            if route or request_model or request_provider:
                logger.debug(
                    "api_server request selection skipped: session /model override wins for %s",
                    session_key or "")
        elif session_row_model and not confirmed_runtime_lock:
            # An explicit model-only API session selection uses the current provider.
            self._apply_provider_runtime(
                runtime_kwargs, current_provider, target_model=session_row_model)
            model = resolve_effective_model(None, session_row_model, model)
            if request_model or request_provider:
                logger.debug(
                    "api_server request selection skipped: session-persisted model wins for %s",
                    session_key or "")
        else:
            # The request's ``model`` selected the route, so its value is the ALIAS — never a
            # model name. Provider-only requests must select that provider's own model.
            effective_model = (route_model or model) if route is not None else (request_model or model)
            effective_provider = request_provider or route_provider or current_provider
            if request_provider and not request_model and not route_model:
                effective_model = self._provider_selection_model(request_provider, model, current_provider)
                if not effective_model:
                    from gateway.platforms.api_server import _ProviderAuthResolutionError
                    raise _ProviderAuthResolutionError("No default model available for selected provider")
            applied = False
            if effective_provider and (bool(request_provider or route_provider) or effective_model != model):
                # A confirmed Browser lock fails closed: never fall through to the previous
                # global provider's credentials.
                applied = self._apply_provider_runtime(
                    runtime_kwargs, effective_provider, target_model=effective_model,
                    required=bool(request_provider) or confirmed_runtime_lock or
                    bool(route_provider and not route_cfg.get("api_key")))
            if not applied and effective_provider and effective_provider != current_provider:
                runtime_kwargs["provider"] = effective_provider
            model = effective_model
            # Per-route explicit transport secrets/base URLs win after provider resolution.
            for key in ("api_key", "base_url", "api_mode"):
                value = _clean_request_string(route_cfg.get(key))
                if value:
                    runtime_kwargs[key] = value
            if route:
                logger.debug(
                    "api_server request selection applied: model=%s provider=%s route_provider=%s request_provider=%s",
                    model, runtime_kwargs.get("provider"), route_provider or "", request_provider or "")
        model = self._recover_or_record_model(model, runtime_kwargs, gateway_session_key)
        return model, session_override, request_model, request_provider
