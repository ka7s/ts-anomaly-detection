from fastapi import FastAPI

from tsad.api.core.settings import settings
from tsad.api.core.logging import configure_logging
from tsad.api.core.middleware import RequestContextMiddleware
from tsad.api.routes.health import router as health_router
from tsad.api.routes.decide import router as decide_router
from tsad.api.routes.decide_many import router as decide_many_router


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title=settings.app_name, version=settings.version)

    app.add_middleware(RequestContextMiddleware)

    app.include_router(health_router)
    app.include_router(decide_router)
    app.include_router(decide_many_router)

    return app


app = create_app()
