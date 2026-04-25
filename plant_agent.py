"""
plant_agent.py — Agrega plantas a Firestore con imágenes generadas por Gemini.

Uso:
  python plant_agent.py                              # usa plantas_para_agente.csv
  python plant_agent.py plantas_para_agente.csv      # ruta explícita

Formato CSV (encabezado requerido):
  SKU,nombre_limpio,Existencia,Ventas,Popular,variaciones
  PLT-001,Monstera Deliciosa,12,342,si,bolsa 10 litros|maceta 8 pulgadas

Env vars requeridas:
  GEMINI_API_KEY
  FIREBASE_SERVICE_ACCOUNT
  CLOUDINARY_CLOUD_NAME
  CLOUDINARY_API_KEY
  CLOUDINARY_API_SECRET
  UMBRAL_POPULAR (opcional, default 2000)

Dependencias:
  pip install firebase-admin google-genai python-slugify python-dotenv cloudinary
"""

import csv
import json
import os
import sys
import time
import traceback

from dotenv import load_dotenv
load_dotenv()

from slugify import slugify

import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DEFAULT_CSV = "plantas_para_agente.csv"
LIMIT_ROWS = 5
UMBRAL_POPULAR = int(os.environ.get("UMBRAL_POPULAR", "2000"))

LOG_PATH = "proceso_log.csv"

VALID = {
    "categoria":      {"ornamental", "suculenta", "árbol", "interior", "exterior", "medicinal"},
    "luz":            {"sol directo", "luz indirecta", "media sombra", "sombra"},
    "riego":          {"bajo", "medio", "alto"},
    "cuidado":        {"fácil", "intermedio", "difícil"},
    "mascotas":       {"tóxica", "no tóxica"},
    "disponibilidad": {"disponible"},
}

SYSTEM_PROMPT = """Eres un experto botánico hispanohablante.
Cuando te pida información sobre una planta, investígala con Google Search y devuelve
ÚNICAMENTE un objeto JSON válido (sin markdown, sin texto extra) con estos campos:

{
  "identificada": true | false,
  "nombreCientifico": "string",
  "descripcion": "string — 2 o 3 oraciones descriptivas en español",
  "categoria": "ornamental | suculenta | árbol | interior | exterior | medicinal",
  "luz": "sol directo | luz indirecta | media sombra | sombra",
  "riego": "bajo | medio | alto",
  "cuidado": "fácil | intermedio | difícil",
  "mascotas": "tóxica | no tóxica",
  "etiquetas": ["array de strings en español, 3-6 etiquetas relevantes"]
}

Reglas:
- Si NO puedes identificar con confianza la planta por el nombre dado, o el nombre
  es ambiguo o no corresponde a una planta real, devuelve:
  {"identificada": false, "razon": "explicación breve"}
- Usa exactamente los valores del enum para cada campo.
- No inventes datos: si dudas, devuelve identificada=false.
"""

CLOUDINARY_FOLDER = "plantas"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_plants_from_csv(path: str) -> list[dict]:
    plants = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for i, row in enumerate(reader, start=2):
            sku = (row.get("SKU") or "").strip()
            nombre = (row.get("nombre_limpio") or "").strip()
            if not sku or not nombre:
                print(f"  [WARN] Fila {i} ignorada: SKU o nombre_limpio vacío")
                continue
            try:
                ventas = int(float((row.get("Ventas") or "0").strip() or "0"))
            except ValueError:
                ventas = 0
            popular_raw = (row.get("Popular") or "").strip().lower()
            popular = popular_raw in ("si", "sí", "yes", "true", "1")
            variaciones_raw = (row.get("variaciones") or "").strip()
            variaciones = [v.strip() for v in variaciones_raw.split("|") if v.strip()]
            plants.append({
                "sku": sku,
                "nombre": nombre,
                "ventas": ventas,
                "popular": popular,
                "variaciones": variaciones,
            })
    return plants


# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------

def init_clients():
    required = ("FIREBASE_SERVICE_ACCOUNT", "GEMINI_API_KEY",
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
    gem = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return db, gem


# ---------------------------------------------------------------------------
# Investigación con Gemini 1.5 Flash + Google Search grounding
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON en respuesta: {text[:200]}")
    return json.loads(text[start:end + 1])


def research_plant(nombre: str, gem: genai.Client) -> tuple[dict, int, int]:
    """Investiga planta con Gemini 1.5 Flash + Google Search grounding.
    Devuelve (data, input_tokens, output_tokens). data puede tener identificada=False."""
    print(f"  -> Investigando '{nombre}' con Gemini 1.5 Flash + Google Search ...")

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Investiga la planta llamada '{nombre}' y devuelve el JSON solicitado."
    )

    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())],
        temperature=0.2,
    )

    response = gem.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
        config=config,
    )

    in_tokens = 0
    out_tokens = 0
    if getattr(response, "usage_metadata", None):
        in_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        out_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    text_chunks = []
    for cand in response.candidates or []:
        for part in (cand.content.parts if cand.content else []):
            if getattr(part, "text", None):
                text_chunks.append(part.text)
    full_text = "\n".join(text_chunks).strip()
    if not full_text:
        raise RuntimeError(f"Gemini no devolvió texto para '{nombre}'")

    return _extract_json(full_text), in_tokens, out_tokens


def validate_plant_data(data: dict, nombre: str) -> dict:
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
            print(f"    [WARN] Campo '{campo}' inválido: '{val}' -> '{defaults[campo]}'")
            data[campo] = defaults[campo]

    if not isinstance(data.get("etiquetas"), list):
        data["etiquetas"] = [nombre.lower()]
    return data


# ---------------------------------------------------------------------------
# Imagen con Gemini + Cloudinary
# ---------------------------------------------------------------------------

def pick_container(variaciones: list[str]) -> str:
    """Devuelve 'maceta' si hay alguna variación maceta, else 'bolsa'."""
    for v in variaciones:
        if "maceta" in v.lower():
            return "maceta"
    return "bolsa"


def generate_and_upload_image(nombre_cientifico: str, container: str,
                              public_id: str, gem: genai.Client) -> str:
    if container == "maceta":
        prompt = (
            f"professional botanical photo of {nombre_cientifico} in a plain black "
            "plastic nursery pot, white background, studio lighting, high quality, "
            "no decorations, no patterns"
        )
    else:
        prompt = (
            f"professional botanical photo of {nombre_cientifico} in a plain black "
            "plastic nursery bag, white background, studio lighting, high quality, "
            "no decorations, no patterns"
        )

    print(f"  -> Generando imagen ({container}) con Gemini ...")
    response = gem.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    image_bytes = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_bytes = part.inline_data.data
            break

    if not image_bytes:
        raise ValueError("Gemini no devolvió imagen")

    print(f"  -> Subiendo a Cloudinary: {CLOUDINARY_FOLDER}/{public_id} ...")
    result = cloudinary.uploader.upload(
        image_bytes,
        folder=CLOUDINARY_FOLDER,
        public_id=public_id,
        resource_type="image",
    )
    return result["secure_url"]


# ---------------------------------------------------------------------------
# Firestore
# ---------------------------------------------------------------------------

def save_to_firestore(plant: dict, plant_data: dict, imagen_url: str | None, db) -> str:
    sku = plant["sku"]
    doc = {
        "sku":              sku,
        "nombre":           plant["nombre"],
        "nombreCientifico": plant_data.get("nombreCientifico", ""),
        "descripcion":      plant_data.get("descripcion", ""),
        "categoria":        plant_data["categoria"],
        "luz":              plant_data["luz"],
        "riego":            plant_data["riego"],
        "cuidado":          plant_data["cuidado"],
        "mascotas":         plant_data["mascotas"],
        "disponibilidad":   "disponible",
        "vistas":           0,
        "etiquetas":        plant_data.get("etiquetas", []),
        "variaciones":      plant["variaciones"],
        "imagenes":         [imagen_url] if imagen_url else [],
    }
    print(f"  -> Guardando en Firestore (ID: {sku}) ...")
    db.collection("plantas").document(sku).set(doc)
    return sku


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== plant_agent.py ===\n")

    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] Archivo no encontrado: {csv_path}")

    plants = load_plants_from_csv(csv_path)
    print(f"Cargadas {len(plants)} plantas desde '{csv_path}'")

    if LIMIT_ROWS:
        plants = plants[:LIMIT_ROWS]
        print(f"Procesando solo las primeras {len(plants)} (modo prueba)\n")

    db, gem = init_clients()

    log_rows = []
    total_in = 0
    total_out = 0
    ok_count = 0
    skip_count = 0

    for plant in plants:
        sku = plant["sku"]
        nombre = plant["nombre"]
        in_tok = 0
        out_tok = 0
        status = "saltada"
        razon = ""

        print(f"\n[{sku}] {nombre}")

        try:
            # 1. Investigar
            try:
                raw_data, in_tok, out_tok = research_plant(nombre, gem)
            except Exception as e:
                razon = f"investigación falló: {e}"
                raise

            if not raw_data.get("identificada", True):
                razon = f"no identificada: {raw_data.get('razon', 'sin razón')}"
                print(f"  [SKIP] {razon}")
            else:
                plant_data = validate_plant_data(raw_data, nombre)

                # Etiqueta popular: por columna Popular del CSV
                if plant["popular"]:
                    if "popular" not in plant_data["etiquetas"]:
                        plant_data["etiquetas"].append("popular")

                # 2. Imagen
                container = pick_container(plant["variaciones"])
                imagen_url = None
                try:
                    public_id = slugify(sku)
                    imagen_url = generate_and_upload_image(
                        plant_data["nombreCientifico"], container, public_id, gem
                    )
                except Exception as e:
                    razon = f"imagen falló: {e}"
                    print(f"  [SKIP planta] {razon}")
                    raise

                # 3. Firestore
                try:
                    save_to_firestore(plant, plant_data, imagen_url, db)
                except Exception as e:
                    razon = f"firestore falló: {e}"
                    raise

                status = "ok"
                ok_count += 1
                print(f"  [OK] plantas/{sku}")

        except Exception as e:
            if not razon:
                razon = f"error inesperado: {e}"
            traceback.print_exc()

        if status != "ok":
            skip_count += 1

        total_in += in_tok
        total_out += out_tok
        print(f"[{sku}] {nombre} -> {status} | tokens entrada: {in_tok} | tokens salida: {out_tok}")

        log_rows.append({
            "SKU": sku,
            "nombre": nombre,
            "status": status,
            "razon": razon,
            "tokens_entrada": in_tok,
            "tokens_salida": out_tok,
        })

        if plant != plants[-1]:
            time.sleep(2)

    # Log CSV
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["SKU", "nombre", "status", "razon",
                                                "tokens_entrada", "tokens_salida"])
        writer.writeheader()
        writer.writerows(log_rows)

    # Resumen
    cost = (total_in / 1_000_000) * 0.075 + (total_out / 1_000_000) * 0.30
    print("\n=== Resumen ===")
    print(f"Total procesadas: {len(plants)}")
    print(f"Exitosas:         {ok_count}")
    print(f"Saltadas:         {skip_count}")
    print(f"Tokens entrada:   {total_in}")
    print(f"Tokens salida:    {total_out}")
    print(f"Costo estimado:   ${cost:.6f} USD (Gemini 1.5 Flash, sin grounding fees)")
    print(f"Log guardado:     {LOG_PATH}")


if __name__ == "__main__":
    main()
