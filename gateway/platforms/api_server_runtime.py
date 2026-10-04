"""Session/request route selection for the API-server adapter."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("gateway.platforms.api_server")


class SessionRuntimeMixin:
    """Keep explicit session choices separate from the last executed model."""

    def _stored_session_model(self, session: Any) -> Optional[str]:
        if not isinstance(session, dict):
            return None
        stored = session.get("model")
        lock = self._parse_session_model_config(session.get("model_config")).get("browser_model_lock")
        # API creation records explicit selections here (confirmed or not). The row's model
        # alone is execution metadata and must not pin a previous default after a restart.
        if isinstance(lock, dict):
            # Confirmed locks are handled by the API prelude. An explicit one-turn request
            # may bypass one without replacing it; don't reapply it as a raw stored model.
            if lock.get("confirmed"):
                return None
            stored = lock.get("model")
        elif not self._resolve_route(stored):
            return None
        if not stored or stored == self._model_name:
            return None
        return stored

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
                # A native API caller may address a transcript without a gateway key.
                # IDs can repeat across independent profile DBs; only its owning home may pin it.
                request_home = Path(get_hermes_home()).resolve()
                for entry in store.list_sessions():
                    if entry.session_id != session_key:
                        continue
                    owner_home = store._profile_home_for_key(entry.session_key)
                    if owner_home is None and not store._named_profile_for_key(entry.session_key):
                        owner_home = store._routing_home
                    if owner_home is not None and Path(owner_home).resolve() == request_home:
                        session_key = entry.session_key
                        break
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
            provider_runtime = session_override if session_override.get("api_key") else self._resolve_provider_runtime(
                _clean_request_string(session_override.get("provider")) or current_provider,
                target_model=override_model, required=False)
            if provider_runtime:
                model = override_model
                _apply_runtime_agent_overrides(runtime_kwargs, provider_runtime)
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
            # model name; a route with no ``model`` key keeps the global default.
            effective_model = (route_model or model) if route is not None else (request_model or model)
            effective_provider = request_provider or route_provider or current_provider
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
            for key in ("api_key", "base_url"):
                value = _clean_request_string(route_cfg.get(key))
                if value:
                    runtime_kwargs[key] = value
            if route:
                logger.debug(
                    "api_server request selection applied: model=%s provider=%s route_provider=%s request_provider=%s",
                    model, runtime_kwargs.get("provider"), route_provider or "", request_provider or "")
        model = self._recover_or_record_model(model, runtime_kwargs, gateway_session_key)
        return model, session_override, request_model, request_provider
