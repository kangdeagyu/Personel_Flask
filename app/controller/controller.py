from flask_restx import Namespace
from ..routes.ai_image_routes import ai_routes


def register_namespaces(api):
    ai_ns = Namespace("ai")

    ai_routes(ai_ns)

    api.add_namespace(ai_ns)