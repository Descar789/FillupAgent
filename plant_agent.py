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
LIMIT_ROWS = None  # None = procesar todas
CHECKPOINT_EVERY = 10
UMBRAL_POPULAR = int(os.environ.get("UMBRAL_POPULAR", "2000"))

LOG_PATH = "proceso_log.csv"

LOG_FIELDS = [
    "indice", "SKU", "nombre", "status", "paso", "razon",
    "error_kind", "tokens_entrada", "tokens_salida", "traceback",
]

# error_kind: "logical" no reintentable, "transient" reintentable, "" cuando ok
TRANSIENT_HINTS = (
    "timeout", "deadline", "unavailable", "connection", "connection error",
    "overload", "503", "429", "5xx", "cloudinary", "upload failed",
    "firebase", "firestore", "ssl", "reset", "temporarily", "internal error",
    "imagen falló", "gemini no devolvió",
)
LOGICAL_HINTS = (
    "no identificada", "ambiguo", "ambigu", "sin nombre científico",
    "sin nombre cientifico", "category mismatch", "categoría inválida",
    "categoria invalida", "planta diferente", "confusión", "confusion",
)


def classify_error(razon: str) -> str:
    r = razon.lower()
    for h in LOGICAL_HINTS:
        if h in r:
            return "logical"
    for h in TRANSIENT_HINTS:
        if h in r:
            return "transient"
    return "transient"  # default: reintentar errores desconocidos

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
    print(f"  -> Investigando '{nombre}' con Gemini 2.5 Flash + Google Search ...")

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Investiga la planta llamada '{nombre}' y devuelve el JSON solicitado."
    )

    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.2,
    )

    response = gem.models.generate_content(
        model="gemini-2.5-flash",
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


def _build_image_prompts(nombre_cientifico: str, container: str) -> list[str]:
    receptaculo = "plastic nursery pot" if container == "maceta" else "plastic nursery bag"
    primary = (
        f"professional botanical photo of {nombre_cientifico} in a plain black "
        f"{receptaculo}, white background, studio lighting, high quality, "
        "no decorations, no patterns"
    )
    fallback = (
        f"simple stock photo of a healthy {nombre_cientifico} plant placed in a "
        f"black {receptaculo}, neutral seamless white studio background, soft "
        "natural lighting, centered composition, realistic, no text, no watermark"
    )
    return [primary, fallback]


def _try_generate_image(prompt: str, gem: genai.Client) -> tuple[bytes | None, str]:
    """Devuelve (image_bytes, info). info describe por qué falló si bytes es None."""
    response = gem.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    candidates = response.candidates or []
    if not candidates:
        return None, "sin candidates"

    cand = candidates[0]
    finish = getattr(cand, "finish_reason", None)
    if cand.content is None or not getattr(cand.content, "parts", None):
        return None, f"content vacío (finish_reason={finish})"

    for part in cand.content.parts:
        if part.inline_data is not None:
            return part.inline_data.data, "ok"

    return None, f"sin inline_data (finish_reason={finish})"


def generate_and_upload_image(nombre_cientifico: str, container: str,
                              public_id: str, gem: genai.Client) -> str:
    prompts = _build_image_prompts(nombre_cientifico, container)
    image_bytes = None
    last_info = ""

    for i, prompt in enumerate(prompts, start=1):
        label = "principal" if i == 1 else "fallback"
        print(f"  -> Generando imagen ({container}, intento {i}/{len(prompts)} {label}) con Gemini ...")
        try:
            image_bytes, info = _try_generate_image(prompt, gem)
        except Exception as e:
            last_info = f"excepción intento {i}: {e}"
            print(f"    [WARN] {last_info}")
            continue

        if image_bytes:
            break
        last_info = f"intento {i}: {info}"
        print(f"    [WARN] {last_info}")

    if not image_bytes:
        raise ValueError(f"Gemini no devolvió imagen tras {len(prompts)} intentos. {last_info}")

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

def migrate_old_log(path: str, output: str | None = None) -> None:
    """Si log no tiene 'error_kind', lo migra in-place agregando columna clasificada."""
    if not os.path.isfile(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        first = f.readline()
    if "error_kind" in first:
        if output and output != path:
            os.replace(path, output)
        return

    rows_out = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip()
            razon = row.get("razon") or ""
            # Migrar status legacy
            if status == "error":
                status = "saltada"
            if status == "sin_imagen":
                status = "saltada"
                if not razon:
                    razon = "imagen falló (legacy sin_imagen)"
            kind = ""
            if status == "saltada":
                kind = classify_error(razon)
            rows_out.append({
                "indice":         row.get("indice", ""),
                "SKU":            row.get("SKU", ""),
                "nombre":         row.get("nombre", ""),
                "status":         status,
                "paso":           row.get("paso", ""),
                "razon":          razon,
                "error_kind":     kind,
                "tokens_entrada": row.get("tokens_entrada", "0"),
                "tokens_salida":  row.get("tokens_salida", "0"),
                "traceback":      row.get("traceback", ""),
            })

    target = output or path
    with open(target, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[MIGRACIÓN] {path} -> {target}: {len(rows_out)} filas migradas con error_kind")


def load_existing_log(path: str) -> dict[str, dict]:
    """Devuelve dict sku -> última fila registrada."""
    if not os.path.isfile(path):
        return {}
    last_by_sku: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = (row.get("SKU") or "").strip()
            if sku:
                last_by_sku[sku] = row
    return last_by_sku


def should_skip_existing(entry: dict) -> tuple[bool, str]:
    """Devuelve (skip, motivo)."""
    if not entry:
        return False, ""
    status = entry.get("status", "")
    if status == "ok":
        return True, "ya procesada (ok)"
    if status == "saltada":
        kind = entry.get("error_kind", "")
        if kind == "logical":
            return True, f"saltada lógica previa: {entry.get('razon', '')[:80]}"
    return False, ""


def firestore_doc_exists(db, sku: str) -> bool:
    try:
        snap = db.collection("plantas").document(sku).get()
        return snap.exists
    except Exception as e:
        print(f"  [WARN] no se pudo verificar Firestore para {sku}: {e}")
        return False


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

    # Migrar log viejo si schema cambió (in-place, conservar datos)
    migrate_old_log(LOG_PATH)
    # Si existe .bak de migración previa fallida, fusionar al log actual
    bak_path = LOG_PATH + ".bak"
    if os.path.isfile(bak_path) and not os.path.isfile(LOG_PATH):
        print(f"[MIGRACIÓN] Recuperando datos de {bak_path} -> {LOG_PATH}")
        migrate_old_log(bak_path, output=LOG_PATH)

    # Reanudación: leer log previo
    existing = load_existing_log(LOG_PATH)
    if existing:
        ya_ok = sum(1 for r in existing.values() if r.get("status") == "ok")
        ya_logical = sum(1 for r in existing.values()
                         if r.get("status") == "saltada" and r.get("error_kind") == "logical")
        ya_retry = sum(1 for r in existing.values()
                       if r.get("status") == "saltada" and r.get("error_kind") != "logical")
        ya_retry += sum(1 for r in existing.values() if r.get("status") == "error")
        pendientes_total = len(plants) - ya_ok - ya_logical
        print(f"Reanudando proceso desde '{LOG_PATH}'...")
        print(f"  Ya procesadas (ok):           {ya_ok}")
        print(f"  Saltadas lógicas (no retry):  {ya_logical}")
        print(f"  Reintentos pendientes:        {ya_retry}")
        print(f"  Pendientes totales:           {max(0, pendientes_total)}\n")
    else:
        print("No hay log previo. Empezando desde cero.\n")

    db, gem = init_clients()

    # Log append-only
    nuevo = not os.path.isfile(LOG_PATH)
    log_file = open(LOG_PATH, "a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if nuevo:
        log_writer.writeheader()
        log_file.flush()

    def write_log(row):
        full = {k: row.get(k, "") for k in LOG_FIELDS}
        log_writer.writerow(full)
        log_file.flush()

    total_in = 0
    total_out = 0
    ok_count = 0
    skip_logical_count = 0
    skip_transient_count = 0
    skipped_already = 0
    skipped_firestore = 0

    for idx, plant in enumerate(plants, start=1):
        sku = plant["sku"]
        nombre = plant["nombre"]
        print(f"\n[{idx}/{len(plants)}] [{sku}] {nombre}")

        # 1. Verificar log previo
        prev = existing.get(sku)
        skip, motivo = should_skip_existing(prev)
        if skip:
            skipped_already += 1
            print(f"  [SKIP] {motivo}")
            continue

        # 2. Verificar Firestore
        if firestore_doc_exists(db, sku):
            print(f"  [SKIP] ya existe en Firestore — registrando en log y saltando")
            write_log({
                "indice": idx, "SKU": sku, "nombre": nombre,
                "status": "ok", "paso": "", "razon": "ya existía en Firestore",
                "error_kind": "", "tokens_entrada": 0, "tokens_salida": 0,
                "traceback": "",
            })
            skipped_firestore += 1
            ok_count += 1
            continue

        # 3. Procesar
        in_tok = 0
        out_tok = 0
        status = "saltada"
        paso = ""
        razon = ""
        tb_str = ""
        error_kind = ""

        try:
            # Investigar
            paso = "investigacion"
            try:
                raw_data, in_tok, out_tok = research_plant(nombre, gem)
            except Exception as e:
                razon = f"investigación falló: {e}"
                raise

            if not raw_data.get("identificada", True):
                paso = "identificacion"
                razon = f"no identificada: {raw_data.get('razon', 'sin razón')}"
                error_kind = "logical"
                print(f"  [SKIP lógico] {razon}")
            else:
                # Validar nombre científico
                nombre_cientifico = (raw_data.get("nombreCientifico") or "").strip()
                if not nombre_cientifico or len(nombre_cientifico) < 3:
                    paso = "validacion"
                    razon = "sin nombre científico confiable"
                    error_kind = "logical"
                    raise ValueError(razon)

                # Validar categoría
                cat = (raw_data.get("categoria") or "").strip()
                if cat and cat not in VALID["categoria"]:
                    paso = "validacion"
                    razon = f"categoría inválida devuelta: '{cat}'"
                    error_kind = "logical"
                    raise ValueError(razon)

                paso = "validacion"
                plant_data = validate_plant_data(raw_data, nombre)

                if plant["popular"] and "popular" not in plant_data["etiquetas"]:
                    plant_data["etiquetas"].append("popular")

                # Imagen
                paso = "imagen"
                container = pick_container(plant["variaciones"])
                try:
                    public_id = slugify(sku)
                    imagen_url = generate_and_upload_image(
                        plant_data["nombreCientifico"], container, public_id, gem
                    )
                except Exception as e:
                    razon = f"imagen falló: {e}"
                    raise

                # Firestore
                paso = "firestore"
                try:
                    save_to_firestore(plant, plant_data, imagen_url, db)
                except Exception as e:
                    razon = f"firestore falló: {e}"
                    raise

                status = "ok"
                paso = ""
                ok_count += 1
                print(f"  [OK] plantas/{sku}")

        except Exception as e:
            status = "saltada"
            if not razon:
                razon = f"error inesperado: {e}"
            if not error_kind:
                error_kind = classify_error(razon)
            tb_str = traceback.format_exc().replace("\n", " | ")
            print(f"  [SKIP {error_kind}] paso={paso} razon={razon}")

        if status == "saltada":
            if error_kind == "logical":
                skip_logical_count += 1
            else:
                skip_transient_count += 1

        total_in += in_tok
        total_out += out_tok

        write_log({
            "indice": idx, "SKU": sku, "nombre": nombre,
            "status": status, "paso": paso, "razon": razon,
            "error_kind": error_kind,
            "tokens_entrada": in_tok, "tokens_salida": out_tok,
            "traceback": tb_str,
        })

        if idx % CHECKPOINT_EVERY == 0 and idx < len(plants):
            print(f"\n--- CHECKPOINT {idx}/{len(plants)}: "
                  f"{ok_count} ok, {skip_logical_count} lógicas, "
                  f"{skip_transient_count} transient, {skipped_already} ya hechas ---\n")

        if plant != plants[-1]:
            time.sleep(2)

    log_file.close()

    cost = (total_in / 1_000_000) * 0.30 + (total_out / 1_000_000) * 2.50
    print("\n=== Resumen ===")
    print(f"Total filas CSV:                {len(plants)}")
    print(f"Saltadas por log previo:        {skipped_already}")
    print(f"Saltadas por Firestore previo:  {skipped_firestore}")
    print(f"Procesadas ok esta corrida:     {ok_count - skipped_firestore}")
    print(f"Saltadas lógicas (no retry):    {skip_logical_count}")
    print(f"Saltadas transient (retry):     {skip_transient_count}")
    print(f"Tokens entrada:   {total_in}")
    print(f"Tokens salida:    {total_out}")
    print(f"Costo estimado:   ${cost:.6f} USD (Gemini 2.5 Flash)")
    print(f"Log:              {LOG_PATH}")


if __name__ == "__main__":
    main()
