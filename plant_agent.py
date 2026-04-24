"""
plant_agent.py — Agrega 5 plantas a Firestore con imágenes generadas por Gemini.

Env vars requeridas:
  ANTHROPIC_API_KEY
  FIREBASE_SERVICE_ACCOUNT  (path al JSON de service account)
  GEMINI_API_KEY

Dependencias:
  pip install firebase-admin anthropic google-genai python-slugify python-dotenv
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from slugify import slugify

import anthropic
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

PLANTS = [
    {"nombre": "Monstera deliciosa", "sku": "PLT-001"},
    {"nombre": "Lavanda",            "sku": "PLT-002"},
    {"nombre": "Cactus San Pedro",   "sku": "PLT-003"},
    {"nombre": "Pothos dorado",      "sku": "PLT-004"},
    {"nombre": "Ficus lyrata",       "sku": "PLT-005"},
]

VARIACIONES_DEFAULT = ["chico", "mediano", "grande"]

VALID = {
    "categoria":   {"ornamental", "suculenta", "árbol", "interior", "exterior", "medicinal"},
    "luz":         {"sol directo", "luz indirecta", "media sombra", "sombra"},
    "riego":       {"bajo", "medio", "alto"},
    "cuidado":     {"fácil", "intermedio", "difícil"},
    "mascotas":    {"tóxica", "no tóxica"},
    "disponibilidad": {"disponible"},
}

SYSTEM_PROMPT = """Eres un experto botánico hispanohablante.
Cuando te pida información sobre una planta, investígala con web_search y devuelve
ÚNICAMENTE un objeto JSON válido (sin markdown, sin texto extra) con estos campos:

{
  "nombreCientifico": "string",
  "descripcion": "string — 2 o 3 oraciones descriptivas en español",
  "categoria": "ornamental | suculenta | árbol | interior | exterior | medicinal",
  "luz": "sol directo | luz indirecta | media sombra | sombra",
  "riego": "bajo | medio | alto",
  "cuidado": "fácil | intermedio | difícil",
  "mascotas": "tóxica | no tóxica",
  "etiquetas": ["array de strings en español, 3-6 etiquetas"]
}

Usa exactamente los valores del enum para cada campo."""

STORAGE_BUCKET = "ornaplant-3ea0c.firebasestorage.app"


# ---------------------------------------------------------------------------
# Inicialización de clientes
# ---------------------------------------------------------------------------

def init_clients():
    for var in ("ANTHROPIC_API_KEY", "FIREBASE_SERVICE_ACCOUNT", "GEMINI_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"[ERROR] Falta variable de entorno: {var}")

    cred = credentials.Certificate(os.environ["FIREBASE_SERVICE_ACCOUNT"])
    firebase_admin.initialize_app(cred, {"storageBucket": STORAGE_BUCKET})

    db = firestore.client()
    ant = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    gem = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    return db, ant, gem


# ---------------------------------------------------------------------------
# Paso 1: Investigar planta con Claude + web_search
# ---------------------------------------------------------------------------

def research_plant(nombre: str, ant: anthropic.Anthropic) -> dict:
    """Llama a Claude con web_search y devuelve dict con datos de la planta."""
    print(f"  -> Investigando '{nombre}' con Claude + web_search ...")

    messages = [
        {
            "role": "user",
            "content": f"Investiga la planta llamada '{nombre}' y devuelve el JSON solicitado.",
        }
    ]

    for attempt in range(5):
        response = ant.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text") and block.text.strip():
                    text = block.text.strip()
                    if text.startswith("```"):
                        text = text.split("```")[1]
                        if text.startswith("json"):
                            text = text[4:]
                    text = text.strip()
                    if text.startswith("{"):
                        return json.loads(text)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": "Continúa y entrega el JSON final ahora.",
            })

        elif response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": "Continúa y entrega el JSON final ahora.",
            })
        else:
            raise ValueError(f"stop_reason inesperado: {response.stop_reason}")

    raise RuntimeError(f"No se pudo obtener datos para '{nombre}' tras varios intentos.")


def validate_plant_data(data: dict, nombre: str) -> dict:
    """Valida y corrige campos con valores inválidos."""
    defaults = {
        "categoria": "ornamental",
        "luz":       "luz indirecta",
        "riego":     "medio",
        "cuidado":   "intermedio",
        "mascotas":  "no tóxica",
    }
    for campo, validos in VALID.items():
        if campo == "disponibilidad":
            data["disponibilidad"] = "disponible"
            continue
        val = data.get(campo, "")
        if val not in validos:
            print(f"    [WARN] Campo '{campo}' inválido: '{val}' -> usando '{defaults[campo]}'")
            data[campo] = defaults[campo]

    if not isinstance(data.get("etiquetas"), list):
        data["etiquetas"] = [nombre.lower()]

    return data


# ---------------------------------------------------------------------------
# Paso 2: Generar imagen con Gemini y subir a Firebase Storage
# ---------------------------------------------------------------------------

def generate_and_upload_image(nombre: str, nombre_cientifico: str, slug: str, gem: genai.Client) -> str:
    """Genera imagen con Gemini, sube a Firebase Storage y devuelve la URL pública."""
    print(f"  -> Generando imagen con Gemini: '{nombre_cientifico}' ...")

    prompt = (
        f"High quality botanical photograph of {nombre_cientifico} ({nombre}), "
        "studio lighting, white background, sharp focus, professional plant nursery photo."
    )

    response = gem.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    image_bytes = None
    mime_type = "image/jpeg"
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_bytes = part.inline_data.data
            mime_type = part.inline_data.mime_type or "image/jpeg"
            break

    if not image_bytes:
        raise ValueError(f"Gemini no devolvió imagen para '{nombre}'")

    ext = "jpg" if "jpeg" in mime_type else mime_type.split("/")[-1]
    blob_path = f"plantas/{slug}.{ext}"

    print(f"  -> Subiendo a Firebase Storage: {blob_path} ...")
    bucket = storage.bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_string(image_bytes, content_type=mime_type)
    blob.make_public()

    return blob.public_url


# ---------------------------------------------------------------------------
# Paso 3: Guardar documento en Firestore
# ---------------------------------------------------------------------------

def save_to_firestore(nombre: str, sku: str, plant_data: dict, imagen_url: str, db) -> str:
    """Crea documento en colección 'plantas' y devuelve el ID del documento."""
    nombre_slug = slugify(nombre)

    doc = {
        "nombre":           nombre,
        "nombreCientifico": plant_data.get("nombreCientifico", ""),
        "sku":              sku,
        "descripcion":      plant_data.get("descripcion", ""),
        "categoria":        plant_data["categoria"],
        "luz":              plant_data["luz"],
        "riego":            plant_data["riego"],
        "cuidado":          plant_data["cuidado"],
        "mascotas":         plant_data["mascotas"],
        "disponibilidad":   "disponible",
        "sucursal":         "ambas",
        "vistas":           0,
        "etiquetas":        plant_data.get("etiquetas", []),
        "variaciones":      VARIACIONES_DEFAULT,
        "imagenes":         [imagen_url],
    }

    print(f"  -> Guardando en Firestore (ID: {nombre_slug}) ...")
    db.collection("plantas").document(nombre_slug).set(doc)
    return nombre_slug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== plant_agent.py ===\n")
    db, ant, gem = init_clients()

    results = []

    for plant in PLANTS:
        nombre = plant["nombre"]
        sku = plant["sku"]
        print(f"\n[{sku}] Procesando: {nombre}")

        try:
            # 1. Investigar
            raw_data = research_plant(nombre, ant)
            plant_data = validate_plant_data(raw_data, nombre)
            print(f"    Categoria: {plant_data['categoria']} | Luz: {plant_data['luz']} | Riego: {plant_data['riego']}")

            # 2. Generar imagen con Gemini y subir a Firebase Storage
            nombre_slug = slugify(nombre)
            imagen_url = generate_and_upload_image(nombre, plant_data["nombreCientifico"], nombre_slug, gem)
            print(f"    URL imagen: {imagen_url}")

            # 3. Guardar en Firestore
            doc_id = save_to_firestore(nombre, sku, plant_data, imagen_url, db)
            print(f"    OK Documento creado: plantas/{doc_id}")

            results.append({"nombre": nombre, "sku": sku, "doc_id": doc_id, "ok": True})

        except Exception as e:
            print(f"    [ERROR] {e}")
            results.append({"nombre": nombre, "sku": sku, "ok": False, "error": str(e)})

        if plant != PLANTS[-1]:
            time.sleep(2)

    print("\n=== Resumen ===")
    for r in results:
        estado = "OK" if r["ok"] else "XX"
        print(f"  {estado} {r['sku']} {r['nombre']}" + (f" -> plantas/{r['doc_id']}" if r["ok"] else f" -> {r.get('error', '')}"))


if __name__ == "__main__":
    main()
