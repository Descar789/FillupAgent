"""
plant_agent.py — Agrega plantas a Firestore con imágenes generadas por Gemini.

Uso:
  python plant_agent.py                  # usa lista hardcodeada (5 plantas)
  python plant_agent.py plantas.csv      # lee desde CSV/TXT

Formato CSV (encabezado requerido):
  nombre,sku,ventas,variaciones
  Monstera deliciosa,PLT-001,342,bolsa 10 litros|maceta 8 pulgadas

Env vars requeridas:
  ANTHROPIC_API_KEY
  FIREBASE_SERVICE_ACCOUNT  (path al JSON de service account)
  GEMINI_API_KEY

Dependencias:
  pip install firebase-admin anthropic google-genai python-slugify python-dotenv cloudinary
"""

import csv
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from slugify import slugify

import cloudinary
import cloudinary.uploader
import anthropic
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

UMBRAL_POPULAR = 200  # plantas con ventas > este valor reciben etiqueta "popular"

PLANTS_DEFAULT = [
    {"nombre": "Monstera deliciosa", "sku": "PLT-001", "ventas": 342, "variaciones": ["bolsa 10 litros", "maceta 8 pulgadas"]},
    {"nombre": "Lavanda",            "sku": "PLT-002", "ventas":  89, "variaciones": ["bolsa 10 litros", "maceta 8 pulgadas"]},
    {"nombre": "Cactus San Pedro",   "sku": "PLT-003", "ventas": 201, "variaciones": ["bolsa 10 litros", "maceta 8 pulgadas"]},
    {"nombre": "Pothos dorado",      "sku": "PLT-004", "ventas": 415, "variaciones": ["bolsa 10 litros", "maceta 8 pulgadas"]},
    {"nombre": "Ficus lyrata",       "sku": "PLT-005", "ventas":  54, "variaciones": ["bolsa 10 litros", "maceta 8 pulgadas"]},
]

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

CLOUDINARY_FOLDER = "plantas"


def load_plants_from_csv(path: str) -> list[dict]:
    """Lee plantas desde CSV con columnas: nombre, sku, ventas, variaciones."""
    plants = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for i, row in enumerate(reader, start=2):
            nombre = row.get("nombre", "").strip()
            sku = row.get("sku", "").strip()
            if not nombre or not sku:
                print(f"  [WARN] Fila {i} ignorada: nombre o sku vacío")
                continue
            try:
                ventas = int(row.get("ventas", "0").strip())
            except ValueError:
                print(f"  [WARN] Fila {i} ventas inválidas -> 0")
                ventas = 0
            variaciones_raw = row.get("variaciones", "").strip()
            variaciones = [v.strip() for v in variaciones_raw.split("|") if v.strip()]
            plants.append({"nombre": nombre, "sku": sku, "ventas": ventas, "variaciones": variaciones})
    return plants


# ---------------------------------------------------------------------------
# Inicialización de clientes
# ---------------------------------------------------------------------------

def init_clients():
    required = ("ANTHROPIC_API_KEY", "FIREBASE_SERVICE_ACCOUNT", "GEMINI_API_KEY",
                "CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET")
    for var in required:
        if not os.environ.get(var):
            sys.exit(f"[ERROR] Falta variable de entorno: {var}")

    cred = credentials.Certificate(os.environ["FIREBASE_SERVICE_ACCOUNT"])
    firebase_admin.initialize_app(cred)

    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
    )

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
            model="claude-haiku-4-5-20251001",
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
# Paso 2: Generar imagen con Gemini y subir a Cloudinary
# ---------------------------------------------------------------------------

def generate_and_upload_image(nombre: str, nombre_cientifico: str, slug: str, gem: genai.Client) -> str:
    """Genera imagen con Gemini, sube a Cloudinary y devuelve la URL segura."""
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

    print(f"  -> Subiendo a Cloudinary: {CLOUDINARY_FOLDER}/{slug} ...")
    result = cloudinary.uploader.upload(
        image_bytes,
        folder=CLOUDINARY_FOLDER,
        public_id=slug,
        resource_type="image",
    )
    return result["secure_url"]


# ---------------------------------------------------------------------------
# Paso 3: Guardar documento en Firestore
# ---------------------------------------------------------------------------

def save_to_firestore(nombre: str, sku: str, plant_data: dict, imagen_url: str,
                      variaciones: list, db) -> str:
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
        "variaciones":      variaciones,
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

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.isfile(csv_path):
            sys.exit(f"[ERROR] Archivo no encontrado: {csv_path}")
        plants = load_plants_from_csv(csv_path)
        print(f"Cargadas {len(plants)} plantas desde '{csv_path}'\n")
    else:
        plants = PLANTS_DEFAULT
        print(f"Usando lista por defecto ({len(plants)} plantas)\n")

    db, ant, gem = init_clients()

    results = []

    for plant in plants:
        nombre = plant["nombre"]
        sku = plant["sku"]
        ventas = plant.get("ventas", 0)
        print(f"\n[{sku}] Procesando: {nombre} (ventas: {ventas})")

        try:
            # 1. Investigar
            raw_data = research_plant(nombre, ant)
            plant_data = validate_plant_data(raw_data, nombre)

            # Marcar como popular si supera el umbral
            if ventas > UMBRAL_POPULAR:
                if "popular" not in plant_data["etiquetas"]:
                    plant_data["etiquetas"].append("popular")
                print(f"    popular: si ({ventas} > {UMBRAL_POPULAR})")
            else:
                print(f"    popular: no ({ventas} <= {UMBRAL_POPULAR})")

            print(f"    Categoria: {plant_data['categoria']} | Luz: {plant_data['luz']} | Riego: {plant_data['riego']}")

            # 2. Generar imagen con Gemini y subir a Firebase Storage
            nombre_slug = slugify(nombre)
            imagen_url = generate_and_upload_image(nombre, plant_data["nombreCientifico"], nombre_slug, gem)
            print(f"    URL imagen: {imagen_url}")

            # 3. Guardar en Firestore
            doc_id = save_to_firestore(nombre, sku, plant_data, imagen_url,
                                       plant["variaciones"], db)
            print(f"    OK Documento creado: plantas/{doc_id}")

            results.append({"nombre": nombre, "sku": sku, "doc_id": doc_id, "ok": True})

        except Exception as e:
            print(f"    [ERROR] {e}")
            results.append({"nombre": nombre, "sku": sku, "ok": False, "error": str(e)})

        if plant != plants[-1]:
            time.sleep(2)

    print("\n=== Resumen ===")
    for r in results:
        estado = "OK" if r["ok"] else "XX"
        print(f"  {estado} {r['sku']} {r['nombre']}" + (f" -> plantas/{r['doc_id']}" if r["ok"] else f" -> {r.get('error', '')}"))


if __name__ == "__main__":
    main()
