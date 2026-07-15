"""
=============================================================================
  Smart Farming Agent — Backend (app.py)
  Powered by IBM Watsonx.ai  |  Flask  |  Granite LLM
=============================================================================

AGENT_INSTRUCTIONS
------------------
  Customize the agent behaviour below by editing the variables in this block.

  AGENT_TONE:
      "formal"     — Professional, structured responses.
      "friendly"   — Warm, conversational tone suited for rural farmers.
      "concise"    — Short, bullet-pointed replies (good for SMS-like UX).

  CROP_SPECIALIZATIONS:
      List the crops the agent should have deep expertise in.
      Default: ["cotton", "wheat", "sugarcane", "rice", "soybean", "onion"]

  LANGUAGE_SUPPORT:
      Primary language the agent responds in.
      Options: "English", "Hindi", "Marathi"
      Farmers can also switch language mid-conversation by asking.

  REGION:
      Indian region context for the agent.
      E.g., "Vidarbha, Maharashtra", "Punjab", "Uttar Pradesh"

  SAFETY_RULES:
      - never_dose_without_disclaimer : Always add pesticide safety disclaimer.
      - always_kvk_for_critical      : Recommend Krishi Vigyan Kendra for
                                       critical decisions (disease outbreaks,
                                       crop failure, soil toxicity etc.)
      - no_financial_advice          : Never guarantee market prices.

  MAX_TOKENS / TEMPERATURE:
      Control LLM creativity vs. determinism.
=============================================================================
"""

import os
import base64
import json
import logging
from io import BytesIO

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image

# ── IBM Watsonx ──────────────────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    logging.warning("ibm-watsonx-ai not installed. Running in DEMO mode.")

# =============================================================================
#  ███████╗ AGENT INSTRUCTIONS — EDIT THIS BLOCK ███████╗
# =============================================================================

AGENT_TONE = "friendly"
# Options: "formal" | "friendly" | "concise"

CROP_SPECIALIZATIONS = [
    "cotton", "wheat", "sugarcane", "rice",
    "soybean", "onion", "tur dal", "chickpea"
]

LANGUAGE_SUPPORT = "English"
# Options: "English" | "Hindi" | "Marathi"
# Farmers can ask the agent to switch language at any time.

REGION = "Maharashtra, India"
# Change to your target region: "Punjab", "Uttar Pradesh", "Rajasthan", etc.

SAFETY_RULES = {
    "never_dose_without_disclaimer": True,
    "always_kvk_for_critical": True,
    "no_financial_advice": True,
}

MAX_TOKENS = 1024
TEMPERATURE = 0.7
MODEL_ID = "ibm/granite-13b-instruct-v2"
# Alternative models: "ibm/granite-3-8b-instruct", "ibm/granite-20b-multilingual"

# =============================================================================
#  END OF AGENT INSTRUCTIONS
# =============================================================================

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "smart-farming-dev-key")
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Watsonx client initialisation ────────────────────────────────────────────
_watsonx_model       = None
_watsonx_error       = None   # last connection error message (for diagnostics)
_watsonx_model_used  = None   # which model ID actually connected

# Models tried in order — first available wins
FALLBACK_MODEL_IDS = [
    MODEL_ID,                           # primary (from AGENT_INSTRUCTIONS)
    "ibm/granite-3-8b-instruct",        # smaller, widely available
    "ibm/granite-13b-chat-v2",          # chat variant
    "ibm/granite-7b-lab",               # lab/preview tier
]


def get_watsonx_model():
    global _watsonx_model, _watsonx_error, _watsonx_model_used
    if _watsonx_model is not None:
        return _watsonx_model
    if not WATSONX_AVAILABLE:
        _watsonx_error = "ibm-watsonx-ai SDK not installed. Run: pip install ibm-watsonx-ai"
        return None

    api_key    = os.getenv("IBM_API_KEY", "").strip()
    project_id = os.getenv("IBM_PROJECT_ID", "").strip()
    url        = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()

    if not api_key or api_key == "your_ibm_cloud_api_key_here":
        _watsonx_error = "IBM_API_KEY not set in .env file."
        return None
    if not project_id or project_id == "your_watsonx_project_id_here":
        _watsonx_error = "IBM_PROJECT_ID not set in .env file."
        return None

    try:
        credentials = Credentials(api_key=api_key, url=url)
        client      = APIClient(credentials)
    except Exception as exc:
        _watsonx_error = f"Authentication failed: {exc}"
        logger.error(f"Watsonx auth failed: {exc}")
        return None

    # Try each model ID in order until one succeeds
    last_exc = None
    for mid in FALLBACK_MODEL_IDS:
        try:
            candidate = ModelInference(
                model_id   = mid,
                api_client = client,
                project_id = project_id,
                params     = {
                    GenParams.MAX_NEW_TOKENS:     MAX_TOKENS,
                    GenParams.TEMPERATURE:        TEMPERATURE,
                    GenParams.REPETITION_PENALTY: 1.1,
                },
            )
            # ── Smoke-test: send a minimal prompt to confirm the model responds ──
            test_result = candidate.generate_text(prompt="Hello")
            _ = test_result  # discard result; any non-exception = success
            _watsonx_model      = candidate
            _watsonx_model_used = mid
            _watsonx_error      = None
            logger.info(f"Watsonx model [{mid}] connected and smoke-tested OK.")
            break
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Model [{mid}] failed: {exc}")
            continue

    if _watsonx_model is None:
        _watsonx_error = _format_watsonx_error(last_exc)
        logger.error(f"All Watsonx models failed. Last error: {last_exc}")

    return _watsonx_model


def _format_watsonx_error(exc: Exception | None) -> str:
    """Convert a raw IBM SDK exception into a human-readable diagnosis."""
    if exc is None:
        return "Unknown connection error."
    msg = str(exc).lower()

    if "403" in msg or "forbidden" in msg:
        return (
            "403 Forbidden — The API key does not have permission to use this model. "
            "Go to: IBM Cloud → Watsonx.ai Studio → your Project → Manage → "
            "Access Control → add your service ID/API key with 'Editor' role."
        )
    if "401" in msg or "unauthorized" in msg or "invalid" in msg and "key" in msg:
        return (
            "401 Unauthorized — The IBM_API_KEY is invalid or expired. "
            "Generate a fresh API key at: cloud.ibm.com → Manage → IAM → API keys."
        )
    if "404" in msg or "not found" in msg or "model" in msg:
        return (
            f"Model not found — '{MODEL_ID}' may not be available in your region or plan. "
            "Try changing MODEL_ID in app.py to 'ibm/granite-3-8b-instruct'."
        )
    if "project" in msg:
        return (
            "Project error — IBM_PROJECT_ID may be wrong or the project doesn't exist. "
            "Find it in: Watsonx Studio → your Project → Manage → General → Project ID."
        )
    if "timeout" in msg or "connect" in msg or "network" in msg:
        return (
            "Network timeout — Could not reach IBM_WATSONX_URL. "
            "Check your internet connection and that IBM_WATSONX_URL is correct."
        )
    # Default: return first 300 chars of the raw error
    return f"Connection error: {str(exc)[:300]}"


def reset_watsonx_model():
    """Force re-initialisation (called after .env update)."""
    global _watsonx_model, _watsonx_error, _watsonx_model_used
    _watsonx_model      = None
    _watsonx_error      = None
    _watsonx_model_used = None


# ── System prompt builder ─────────────────────────────────────────────────────
def build_system_prompt(context: dict | None = None) -> str:
    ctx = context or {}
    crop_list  = ", ".join(CROP_SPECIALIZATIONS)
    safety_txt = ""
    if SAFETY_RULES["never_dose_without_disclaimer"]:
        safety_txt += (
            "\n- ALWAYS add this disclaimer before mentioning pesticide quantities: "
            "'⚠️ Pesticide dosages should be verified with a licensed agronomist. "
            "Misuse can harm crops, soil, and health.'"
        )
    if SAFETY_RULES["always_kvk_for_critical"]:
        safety_txt += (
            "\n- For critical decisions (crop failure, unknown disease, soil toxicity), "
            "ALWAYS recommend: 'Please consult your local Krishi Vigyan Kendra (KVK) "
            "for expert guidance.'"
        )
    if SAFETY_RULES["no_financial_advice"]:
        safety_txt += (
            "\n- Never guarantee market prices. Say 'prices are indicative and fluctuate "
            "based on mandi conditions.'"
        )

    tone_map = {
        "formal":   "Use a formal, professional tone.",
        "friendly": "Use a warm, simple, farmer-friendly tone — like an experienced local agronomist helping a neighbour.",
        "concise":  "Be brief and use bullet points. Avoid long paragraphs.",
    }
    tone_instruction = tone_map.get(AGENT_TONE, tone_map["friendly"])

    lang_instruction = ""
    if LANGUAGE_SUPPORT == "Hindi":
        lang_instruction = "Respond in Hindi (Devanagari script) unless the farmer writes in another language."
    elif LANGUAGE_SUPPORT == "Marathi":
        lang_instruction = "Respond in Marathi (Devanagari script) unless the farmer writes in another language."
    else:
        lang_instruction = (
            "Respond in English by default. If the farmer writes in Hindi or Marathi, "
            "respond in the same language."
        )

    # Dynamic context injections
    soil_info    = f"Soil type: {ctx['soil_type']}, pH: {ctx['soil_ph']}. " if ctx.get("soil_type") else ""
    weather_info = f"Current weather: {ctx['weather']}. " if ctx.get("weather") else ""
    crop_info    = f"Farmer's current crop: {ctx['crop']}. " if ctx.get("crop") else ""
    location     = f"Farm location: {ctx.get('location', REGION)}. "

    system_prompt = f"""You are an expert AI Smart Farming Agent specialising in Indian agriculture.

{tone_instruction}
{lang_instruction}

Region context: {REGION}
Your crop expertise includes: {crop_list}.

{location}{soil_info}{weather_info}{crop_info}

Your responsibilities:
1. Personalised crop advisory (sowing, growth stages, harvesting)
2. Fertiliser and irrigation schedules tailored to soil & weather
3. Pest and disease identification + organic & chemical management options
4. Weather-based farming tips (kharif/rabi season awareness)
5. Market price awareness for Indian mandis (APMC rates)
6. Multilingual support (English / Hindi / Marathi)
7. Answer questions about government schemes (PM-KISAN, PMFBY, etc.)

Safety rules:{safety_txt}

Always be empathetic, practical, and grounded in real Indian farming conditions.
When you see an image description, identify the pest/disease and give actionable advice.
End each response with a relevant emoji 🌾 or 🚜 to keep the tone warm.
"""
    return system_prompt


# ── Demo responses (context-aware fallback when Watsonx not configured) ──────
#   Keyed by keyword → response. Matched against lowercased user message.
DEMO_RESPONSES_KEYED = {
    # Fertilizer
    "fertiliz": {
        "wheat":     "💊 **Wheat Fertilizer Schedule (per hectare)**\n\n**Basal dose (at sowing):**\n- Urea: 55 kg (25 kg N)\n- DAP: 110 kg (50 kg P₂O₅)\n- MOP: 50 kg (30 kg K₂O)\n\n**Top dressing:**\n- 1st: 55 kg Urea at Crown Root Initiation (21 DAS)\n- 2nd: 55 kg Urea at Tillering (40 DAS)\n\n> 💡 On black cotton soil with pH 6.5, DAP works well. Zinc deficiency is common — apply 25 kg ZnSO₄/ha if you see interveinal chlorosis.\n\n⚠️ *Pesticide dosages should be verified with a licensed agronomist.*\n\nPlease consult your local **Krishi Vigyan Kendra (KVK), Nagpur** for soil-test-based recommendations. 🌾",
        "cotton":    "💊 **Cotton Fertilizer Schedule (per hectare)**\n\n**Basal dose:**\n- Urea: 55 kg (N: 25 kg)\n- SSP: 375 kg (P₂O₅: 60 kg)\n- MOP: 50 kg (K₂O: 30 kg)\n\n**Top dressing:**\n- 1st (30 DAS): 55 kg Urea\n- 2nd (60 DAS): 55 kg Urea + 25 kg MOP\n\n> 💡 Boron spray (0.2% borax) at flowering improves boll setting.\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "rice":      "💊 **Rice (Paddy) Fertilizer Schedule (per hectare)**\n\n**Basal (before transplanting):**\n- Urea: 45 kg (N: 20 kg)\n- DAP: 110 kg (P₂O₅: 50 kg)\n- MOP: 67 kg (K₂O: 40 kg)\n\n**Top dressing:**\n- 1st (10–15 days after transplant): 90 kg Urea\n- 2nd (Panicle initiation): 45 kg Urea\n\n> 💡 Apply Zinc Sulphate 25 kg/ha as basal on Zinc-deficient soils.\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "sugarcane": "💊 **Sugarcane Fertilizer Schedule (per hectare)**\n\n**Basal:**\n- Urea: 110 kg | SSP: 625 kg | MOP: 100 kg\n\n**Top dressing (split in 3):**\n- 1st (30 DAS): 110 kg Urea\n- 2nd (90 DAS): 110 kg Urea\n- 3rd (150 DAS): 110 kg Urea + 50 kg MOP\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "soybean":   "💊 **Soybean Fertilizer Schedule (per hectare)**\n\nBeing a legume, soybean fixes its own nitrogen via rhizobia.\n\n**Basal dose:**\n- Urea: 20 kg (N: 9 kg starter dose)\n- SSP: 375 kg (P₂O₅: 60 kg)\n- MOP: 50 kg (K₂O: 30 kg)\n\n**Seed treatment:** Rhizobium + PSB culture before sowing.\n\n> 💡 Sulphur 20 kg/ha as gypsum improves oil content.\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "default":   "💊 **General Fertilizer Principles for Indian Crops**\n\n1. Always do a **soil test** (free at KVK / Soil Health Card) before applying fertilizers.\n2. NPK ratio varies by crop — share your crop name for a precise schedule.\n3. **DAP** is the most common phosphorus source; **MOP** for potassium.\n4. Split nitrogen application (2–3 doses) reduces leaching losses.\n5. Micronutrients (Zn, B, Fe) are often deficient in black cotton soils of Vidarbha.\n\nTell me your crop and I'll give you exact quantities! 🌾",
    },
    # Irrigation
    "irrigat": {
        "wheat":     "💧 **Wheat Irrigation Schedule**\n\n| Stage | Days After Sowing | Water (mm) |\n|-------|-----------------|------------|\n| Crown Root Initiation | 21 DAS | 60 mm |\n| Tillering | 40–45 DAS | 60 mm |\n| Jointing | 60–65 DAS | 60 mm |\n| Flowering | 80–85 DAS | 60 mm |\n| Grain Filling | 100–105 DAS | 50 mm |\n| Dough Stage | 115–120 DAS | 50 mm |\n\n> 💡 Total 5–6 irrigations needed. Skip an irrigation if rainfall > 25 mm within 3 days.\n\nAt 28°C current temperature in Nagpur, ensure the soil doesn't dry below field capacity. 🌾",
        "cotton":    "💧 **Cotton Irrigation Schedule**\n\n- Germination (5–7 DAS): Light irrigation if dry sowing\n- Squaring (45–50 DAS): Critical stage — don't miss!\n- Flowering & Boll Development (70–110 DAS): Every 10–15 days\n- Boll Opening (120+ DAS): Stop irrigation\n\n> 💡 Drip irrigation saves 40% water and improves yield by 20–25%.\n\nOn black cotton soil, avoid waterlogging — it causes root rot. 🌾",
        "default":   "💧 **General Irrigation Guidelines**\n\nFor precise irrigation scheduling, please share:\n1. Your crop name\n2. Current growth stage\n3. Soil type\n\n**Quick rule:** Irrigate when soil moisture reaches 50% field capacity. Touch the soil 15 cm deep — if it doesn't clump, it's time to irrigate. 🌾",
    },
    # Pest / disease
    "pest":    {
        "wheat":     "🐛 **Common Wheat Pests & Diseases in Vidarbha**\n\n| Pest/Disease | Symptoms | Management |\n|------------|---------|------------|\n| Aphids | Yellow leaves, sticky honeydew | Spray Imidacloprid 0.5 ml/L water |\n| Rust (Yellow/Brown) | Orange/yellow pustules on leaves | Propiconazole 0.1% spray |\n| Loose Smut | Black powder replacing grains | Seed treatment with Vitavax |\n| Termite | Drying of tillers | Chlorpyrifos soil drench |\n\n⚠️ *Pesticide dosages should be verified with a licensed agronomist. Misuse can harm crops, soil, and health.*\n\nFor unidentified disease spread, please consult **KVK Nagpur**. 🌾",
        "cotton":    "🐛 **Top Cotton Pests in Vidarbha**\n\n| Pest | Symptoms | Management |\n|------|---------|------------|\n| Whitefly | Yellow leaves, sticky deposits | Yellow sticky traps + Thiamethoxam |\n| Pink Bollworm | Entry holes in bolls | Pheromone traps + Cypermethrin |\n| Jassids | Leaf curl, burning edges | Imidacloprid seed treatment |\n| Thrips | Silver streaks on leaves | Spinosad spray |\n\n⚠️ *Pesticide dosages should be verified with a licensed agronomist.*\n\nUpload a photo of your affected plant for more precise identification! 📸 🌾",
        "soybean":   "🐛 **Top Soybean Pests & Diseases in Vidarbha (Kharif)**\n\n| Pest / Disease | Symptoms | Management |\n|---------------|---------|------------|\n| **Girdle Beetle** | Circular cuts on stem, wilting | Remove & destroy affected stems; Profenofos spray |\n| **Whitefly** | Yellow leaves, leaf curl, sooty mold | Yellow sticky traps; Thiamethoxam 25 WG @ 0.3 g/L |\n| **Stem Fly** | Dead heart at seedling stage | Seed treatment with Imidacloprid 70 WS @ 7 g/kg |\n| **Yellow Mosaic Virus (YMV)** | Yellow patches on leaves (viral) | Remove infected plants; control whitefly vector |\n| **Charcoal Rot** | Rotting stem near soil at maturity | Avoid moisture stress; Trichoderma seed treatment |\n| **Pod Borer** | Holes in pods, damaged seeds | Quinalphos 25 EC spray at pod formation |\n\n> ⚠️ **Pesticide dosages should be verified with a licensed agronomist. Misuse can harm crops, soil, and health.**\n\n💡 **Leaf symptoms guide:**\n- Yellow patches → likely **Yellow Mosaic Virus** (no cure — uproot & destroy)\n- Sticky honeydew on leaves → **Whitefly** infestation\n- Ragged leaf edges → **Semilooper** caterpillar\n- Brown lesions on leaves → **Bacterial Pustule** (spray Copper Oxychloride)\n\nUpload a photo 📸 for precise identification. For severe outbreak, consult **KVK Nagpur: 0712-2500849**. 🌾",
        "rice":      "🐛 **Common Rice (Paddy) Pests in India**\n\n| Pest | Symptoms | Management |\n|------|---------|------------|\n| Stem Borer | Dead heart / white ear | Carbofuran 3G in standing water |\n| Brown Plant Hopper | Hopper burn, yellowing from base | Buprofezin spray; avoid excess N |\n| Blast | Diamond-shaped lesions on leaves | Tricyclazole 75 WP spray |\n| Sheath Blight | Water-soaked lesions on leaf sheath | Hexaconazole 5 EC spray |\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "default":   "🐛 **Pest Identification — Tell me more!**\n\nI can see you have a pest or disease concern. To give you the most accurate advice, please tell me:\n\n1. 🌿 **Which crop?** (wheat, soybean, cotton, rice…)\n2. 🍃 **Which part is affected?** (leaves / stem / root / pod / grain)\n3. 📅 **Growth stage?** (seedling / vegetative / flowering / pod filling)\n4. 📊 **How much of your field is affected?** (e.g. 10%, 50%)\n5. 📸 **Upload a photo** using the camera button for instant visual ID!\n\nOr for now, check the **Quick Advisory → Pest & Disease Guide** button on the left panel. 🌾",
    },
    # Market / MSP
    "msp":     {
        "default":   "📊 **Minimum Support Price (MSP) 2024-25 — Key Crops**\n\n| Crop | MSP (₹/Quintal) |\n|------|----------------|\n| Wheat | ₹2,275 |\n| Rice (Common) | ₹2,300 |\n| Cotton (Medium) | ₹7,121 |\n| Soybean | ₹4,892 |\n| Tur Dal | ₹7,550 |\n| Chickpea | ₹5,440 |\n| Onion | Market rate |\n| Sugarcane (FRP) | ₹340/quintal |\n\n> ⚠️ *Prices are indicative and fluctuate based on mandi conditions.*\n\n📱 Check live APMC rates: **agmarknet.gov.in** or call your local **Mandi helpline: 1800-270-0224** 🌾",
    },
    "market":  {
        "default":   "📊 **Minimum Support Price (MSP) 2024-25 — Key Crops**\n\n| Crop | MSP (₹/Quintal) |\n|------|----------------|\n| Wheat | ₹2,275 |\n| Rice (Common) | ₹2,300 |\n| Cotton (Medium) | ₹7,121 |\n| Soybean | ₹4,892 |\n| Tur Dal | ₹7,550 |\n| Chickpea | ₹5,440 |\n| Onion | Market rate |\n| Sugarcane (FRP) | ₹340/quintal |\n\n> ⚠️ *Prices are indicative and fluctuate based on mandi conditions.*\n\n📱 Check live APMC rates: **agmarknet.gov.in** or call **Mandi helpline: 1800-270-0224** 🌾",
    },
    # Schemes
    "scheme":  {
        "default":   "🏛️ **Key Government Schemes for Indian Farmers**\n\n| Scheme | Benefit | How to Apply |\n|--------|---------|-------------|\n| **PM-KISAN** | ₹6,000/year income support | pmkisan.gov.in |\n| **PMFBY** | Crop insurance against natural calamities | pmfby.gov.in |\n| **PM Krishi Sinchai Yojana** | Drip/sprinkler subsidy up to 55% | State Agriculture Dept. |\n| **Soil Health Card** | Free soil test + fertilizer recommendations | Nearest KVK / Agriculture office |\n| **KCC (Kisan Credit Card)** | Low-interest crop loan (4% p.a.) | Nearest bank branch |\n| **e-NAM** | Sell directly on national mandi platform | enam.gov.in |\n\n📞 **Kisan Call Centre:** 1800-180-1551 (Free, 24×7, 22 languages)\n\nVisit your nearest **Krishi Vigyan Kendra (KVK)** for scheme enrollment assistance. 🌾",
    },
    # Weather
    "weather": {
        "default":   "🌡️ **Weather-Based Farming Tips for Vidarbha**\n\nAt **28°C** (your current reading):\n\n✅ **Good conditions for:**\n- Wheat germination (optimal 20–25°C)\n- Pesticide spraying (before 10 AM or after 4 PM)\n- Irrigation in early morning to reduce evaporation\n\n⚠️ **Watch out for:**\n- If temperature rises above 35°C during wheat grain filling — it reduces grain weight\n- Apply mulching to conserve soil moisture\n\n📅 **Nagpur, Vidarbha forecast tip:** Pre-monsoon showers in late May — avoid sowing until soil moisture is stable.\n\nShare your crop name for more specific weather-based advice! 🌾",
    },
    # Sowing
    "sow":     {
        "wheat":     "🌱 **Wheat Sowing Guide — Vidarbha, Maharashtra**\n\n**Best sowing window:** 15 November – 15 December (Rabi)\n\n**Recommended varieties:**\n- GW 322 (drought tolerant, Vidarbha suited)\n- MACS 6222\n- NW 1014\n\n**Seed rate:** 100–125 kg/ha\n**Row spacing:** 22.5 cm\n**Sowing depth:** 5–6 cm\n\n**Seed treatment:**\n1. Thiram 3 g/kg seed (fungicide)\n2. Azotobacter culture 5 g/kg seed\n3. PSB culture 5 g/kg seed\n\n> 💡 Late sowing (after Dec 15) reduces yield by ~1 quintal per week of delay.\n\nFor certified seeds, contact: **Maharashtra State Seeds Corporation** 🌾",
        "cotton":    "🌱 **Cotton Sowing Guide — Vidarbha**\n\n**Best window:** 10 June – 10 July (after 50–75 mm rainfall)\n\n**Varieties:** Bt Cotton hybrids (Bollgard II) are most common.\n**Spacing:** 90 × 60 cm (irrigated) | 120 × 90 cm (rainfed)\n**Seed rate:** 1.5–2 kg/ha (Bt hybrid)\n\n**Seed treatment:**\n- Imidacloprid 70 WS @ 7 g/kg seed for sucking pests\n\n> 💡 Avoid sowing in waterlogged areas — cotton is very sensitive to excess moisture.\n\nConsult **KVK Nagpur** for the best Bt hybrid suited to your village's climate. 🌾",
        "soybean":   "🌱 **Soybean Sowing Guide — Vidarbha, Maharashtra (Kharif)**\n\n**Best sowing window:** 15 June – 15 July\n_(after first good rain of 75–100 mm; soil temperature > 25°C)_\n\n**Recommended Varieties (Vidarbha-suited):**\n| Variety | Duration | Feature |\n|---------|----------|--------|\n| JS 335 | 95–100 days | Most popular, high yield |\n| MACS 450 | 90–95 days | YMV resistant |\n| Phule Agrani | 95 days | Drought tolerant |\n| KDS 344 | 100 days | Wider adaptability |\n\n**Seed rate:** 70–80 kg/ha\n**Row spacing:** 45 cm × 5 cm (plant-to-plant)\n**Sowing depth:** 3–4 cm\n\n**Seed Treatment (in order):**\n1. Thiram + Carbendazim (2:1) @ 3 g/kg seed — fungal protection\n2. Rhizobium japonicum culture @ 5 g/kg seed — nitrogen fixation\n3. PSB culture @ 5 g/kg seed — phosphorus solubilisation\n\n> 💡 **Critical:** Treat seed with Rhizobium culture — it can reduce urea requirement by 80%!\n\n**Weed Management:**\n- Pre-emergence: Pendimethalin 1 kg a.i./ha within 2 days of sowing\n- Post-emergence: Imazethapyr 75 g a.i./ha at 15–20 DAS\n\n> ⚠️ Pesticide dosages should be verified with a licensed agronomist.\n\nFor certified seeds contact: **MSSC** (Maharashtra State Seeds Corporation) or your local KVK. 🌾",
        "rice":      "🌱 **Rice (Paddy) Sowing/Transplanting Guide**\n\n**Nursery sowing:** May 15 – June 15\n**Transplanting:** June 15 – July 15 (25–30 day old seedlings)\n**Seed rate:** 50 kg/ha (transplanted) | 100 kg/ha (direct seeded)\n**Spacing:** 20 × 15 cm\n\n**Seed treatment:**\n1. Soak in water 24 hrs → discard floaters\n2. Carbendazim 2 g/kg (fungal)\n3. Imidacloprid 5 ml/kg (insect)\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "default":   "🌱 **Sowing Advisory**\n\nTell me which crop you'd like to sow and I'll give you:\n- Best sowing dates for your region (Vidarbha)\n- Seed varieties suited to your area\n- Seed rate, spacing and depth\n- Step-by-step seed treatment protocol\n\n*E.g.: \"When should I sow soybean?\" or \"Cotton sowing guide\"* 🌾",
    },
    # Weed management (new category)
    "weed": {
        "soybean":  "🌿 **Soybean Weed Management — Vidarbha**\n\n**Critical period:** First 30–45 days after sowing (DAS) — weeds compete most during this window.\n\n| Herbicide | Type | Dose (a.i./ha) | Timing |\n|-----------|------|---------------|--------|\n| Pendimethalin 30 EC | Pre-emergence | 1.0 kg | Within 2 DAS |\n| Imazethapyr 10 SL | Post-emergence | 75 g | **15–20 DAS** ✅ |\n| Quizalofop-ethyl 5 EC | Post-emergence (grassy) | 50 g | 15–20 DAS |\n| Fomesafen 25 SC | Post-emergence (broad-leaf) | 250 g | 20–25 DAS |\n\n> ✅ **At 20 DAS:** Use **Imazethapyr 10 SL @ 75 g a.i./ha** in 500 L water/ha. This controls both grassy and broad-leaf weeds in soybean.\n\n**Manual option:** Inter-cultivation with bullock-drawn weeder at 20 DAS is highly effective and avoids herbicide cost.\n\n💡 **Tips:**\n- Apply herbicides in moist soil conditions\n- Spray in early morning (before 9 AM) or evening\n- Avoid spraying if rain expected within 4 hours\n- Do NOT use Imazethapyr if you plan to sow wheat next season (carryover)\n\n⚠️ *Herbicide dosages must be verified with a licensed agronomist. Misuse causes crop damage.*\n\nConsult **KVK Nagpur (0712-2500849)** for field-specific recommendations. 🌾",
        "cotton":   "🌿 **Cotton Weed Management**\n\n| Herbicide | Type | Dose | Timing |\n|-----------|------|------|--------|\n| Pendimethalin 38.7 CS | Pre-emergence | 700 g a.i./ha | 0–3 DAS |\n| Pyrithiobac sodium 10 EC | Post-emergence | 62.5 g a.i./ha | 15–25 DAS |\n| Quizalofop-ethyl | Post-emergence (grassy) | 50 g a.i./ha | 15–30 DAS |\n\n**Manual:** Two hand-weedings at 20 and 40 DAS are recommended.\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "wheat":    "🌿 **Wheat Weed Management (Rabi)**\n\n**Critical period:** First 35 days after sowing.\n\n| Herbicide | Target Weeds | Dose | Timing |\n|-----------|-------------|------|--------|\n| Clodinafop 15 WP | Grassy weeds (Phalaris) | 60 g a.i./ha | 30–35 DAS |\n| 2,4-D Amine 58% | Broad-leaf weeds | 500 g a.i./ha | 30–35 DAS |\n| Sulfosulfuron 75 WG | Both types | 25 g a.i./ha | 25–30 DAS |\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "rice":     "🌿 **Rice Weed Management**\n\n| Herbicide | Type | Dose | Timing |\n|-----------|------|------|--------|\n| Pretilachlor 50 EC | Pre-emergence | 500 g a.i./ha | 3–5 DAS |\n| Bispyribac sodium 10 SC | Post-emergence | 25 g a.i./ha | 15–20 DAS |\n| 2,4-D Sodium salt | Broad-leaf | 500 g a.i./ha | 20–25 DAS |\n\n⚠️ *Verify dosages with a licensed agronomist.* 🌾",
        "default":  "🌿 **Weed Management — General Guide**\n\nWeeds reduce crop yield by 20–50% if uncontrolled in the first 30–45 days. Please tell me your crop and I'll give a precise herbicide schedule.\n\n**Common post-emergence herbicides by crop type:**\n- Soybean (15–20 DAS): Imazethapyr 75 g a.i./ha\n- Cotton (15–25 DAS): Pyrithiobac sodium 62.5 g a.i./ha\n- Wheat (30–35 DAS): Clodinafop 60 g a.i./ha\n- Rice (15–20 DAS): Bispyribac sodium 25 g a.i./ha\n\n⚠️ *Dosages must be verified with a licensed agronomist.*\n\nFor best results, consult your local **KVK** for recommended herbicides in your specific soil. 🌾",
    },
    # Crop recommendation (open-ended: "which crop to sow in kharif/rabi")
    "crop_rec": {
        "kharif":   "🌾 **Best Kharif Crops for Vidarbha, Maharashtra**\n\n| Crop | Why suited | Typical yield |\n|------|-----------|---------------|\n| **Soybean** | Black soil, 600–900 mm rain | 12–18 q/ha |\n| **Cotton (Bt)** | Black soil specialty, export crop | 15–20 q/ha seed cotton |\n| **Tur Dal** | Drought-hardy, good MSP | 8–12 q/ha |\n| **Sorghum (Kharif Jowar)** | Dual purpose — grain + fodder | 20–25 q/ha |\n| **Maize** | Short duration, growing demand | 40–60 q/ha |\n| **Green Gram (Moong)** | Short duration, soil-improving | 6–8 q/ha |\n\n> 💡 **Vidarbha recommendation:** Soybean is the #1 kharif crop due to black soil suitability, Rhizobium fixation, and strong MSP (₹4,892/quintal 2024-25). Cotton is #2 for farmers with irrigation.\n\n📅 **Sowing windows:**\n- Soybean: 15 June – 15 July\n- Cotton: 10 June – 10 July\n- Tur Dal: 20 June – 15 July\n\nWhich crop interests you most? I can give you a complete sowing guide! 🌾",
        "rabi":     "🌾 **Best Rabi Crops for Vidarbha, Maharashtra**\n\n| Crop | Why suited | Typical yield |\n|------|-----------|---------------|\n| **Wheat** | Residual moisture in black soil | 25–35 q/ha |\n| **Chickpea (Chana)** | Drought-tolerant legume | 12–18 q/ha |\n| **Linseed** | Oil crop, low water need | 5–8 q/ha |\n| **Safflower** | Drought-hardy, good oil | 8–12 q/ha |\n| **Rabi Onion** | High value vegetable | 200–250 q/ha |\n\n> 💡 After soybean harvest, wheat is the best rabi option — residual nitrogen from Rhizobium reduces fertilizer cost.\n\n📅 **Sowing window:** November 15 – December 15\n\nWhich rabi crop do you want a complete guide for? 🌾",
        "default":  "🌾 **Crop Suitability for Vidarbha (Nagpur Region)**\n\n**Soil:** Predominantly deep black cotton soil (Vertisols) — best for cotton, soybean, tur dal.\n\n**Kharif season (June–October):**\n- 🥇 Soybean — most popular, good MSP\n- 🥈 Cotton (Bt) — high value, export demand\n- 🥉 Tur Dal — drought-hardy, nutritious\n\n**Rabi season (November–March):**\n- 🥇 Wheat — best after soybean\n- 🥈 Chickpea — low-cost, high protein\n- 🥉 Rabi Onion — high-value vegetable\n\n**Ask me for a complete sowing guide for any crop above!**\n\nE.g.: *\"Soybean sowing guide\"* or *\"Which variety of cotton for Nagpur?\"* 🌾",
    },
}

# ── Crop alias normaliser ─────────────────────────────────────────────────────
CROP_ALIASES: dict[str, str] = {
    # Soybean variants
    "soyabean": "soybean", "soya bean": "soybean", "soya": "soybean",
    "soyan": "soybean",
    # Rice / paddy variants
    "paddy": "rice", "dhan": "rice", "chawal": "rice",
    # Cotton variants
    "kapas": "cotton", "bt cotton": "cotton",
    # Wheat variants
    "gehun": "wheat", "gehu": "wheat",
    # Sugarcane variants
    "unkh": "sugarcane", "ganna": "sugarcane",
    # Maize / corn
    "corn": "maize", "makka": "maize",
    # Chickpea / gram
    "gram": "chickpea", "chana": "chickpea", "harbhara": "chickpea",
    # Tur dal variants
    "tur": "tur dal", "arhar": "tur dal", "pigeon pea": "tur dal",
    # Onion
    "kanda": "onion", "pyaj": "onion",
}

def normalise_crop(raw: str) -> str:
    """Normalise crop name via alias map, then return canonical form."""
    r = raw.strip().lower()
    return CROP_ALIASES.get(r, r)


# ── Keyword → category mapping (ordered, highest-priority first) ─────────────
#
#  Priority design:
#   1. Crop recommendation — before "sow" so "which crop in kharif" → crop_rec
#   2. Weed / herbicide    — before "sow" so "herbicide after sowing" → weed
#   3. Sowing guide        — "sow" / "seed rate" / "variety"
#   4. Fertilizer, irrigation, pest, market, schemes, weather
#
KEYWORD_CATEGORY_MAP = [
    # ── Crop recommendation (open-ended season questions) ─────────────────
    ("which crop",         "crop_rec"),
    ("best crop",          "crop_rec"),
    ("suitable crop",      "crop_rec"),
    ("crop suited",        "crop_rec"),
    ("crop for kharif",    "crop_rec"),
    ("crop for rabi",      "crop_rec"),
    ("crop for vidarbha",  "crop_rec"),
    ("what to grow",       "crop_rec"),
    ("what crop",          "crop_rec"),
    ("kon pik",            "crop_rec"),   # Marathi: "which crop"
    ("kaunsi fasal",       "crop_rec"),   # Hindi
    # ── Weed / herbicide — BEFORE "sow" to avoid sowing guide hijack ──────
    ("herbicide",          "weed"),
    ("weedicide",          "weed"),
    ("tarnai nashak",      "weed"),       # Marathi: weedicide
    ("weed control",       "weed"),
    ("weed manag",         "weed"),
    ("post-emergence",     "weed"),
    ("pre-emergence",      "weed"),
    ("imazethapyr",        "weed"),
    ("pendimethalin",      "weed"),
    ("2,4-d",              "weed"),
    ("bispyribac",         "weed"),
    ("clodinafop",         "weed"),
    ("weed",               "weed"),
    # ── Sowing guide ──────────────────────────────────────────────────────
    ("when to sow",        "sow"),
    ("sowing time",        "sow"),
    ("sowing date",        "sow"),
    ("when should i sow",  "sow"),
    ("best time to sow",   "sow"),
    ("seed rate",          "sow"),
    ("seed treatment",     "sow"),
    ("variety",            "sow"),
    ("peryay",             "sow"),        # Hindi: variety
    ("sow",                "sow"),
    ("planting",           "sow"),        # "planting" → sow; NOT bare "plant"
    # ── Fertilizer ────────────────────────────────────────────────────────
    ("fertiliz",           "fertiliz"),
    ("npk",                "fertiliz"),
    ("urea",               "fertiliz"),
    ("dap",                "fertiliz"),
    ("basal dose",         "fertiliz"),
    ("top dress",          "fertiliz"),
    ("manure",             "fertiliz"),
    ("khad",               "fertiliz"),
    ("khaad",              "fertiliz"),
    # ── Irrigation ────────────────────────────────────────────────────────
    ("irrigat",            "irrigat"),
    ("drip",               "irrigat"),
    ("sprinkler",          "irrigat"),
    ("water manag",        "irrigat"),
    ("paani",              "irrigat"),    # Hindi: water
    # ── Pest / disease ────────────────────────────────────────────────────
    ("pesticide",          "pest"),
    ("keetnaashak",        "pest"),
    ("insecticid",         "pest"),
    ("fungicid",           "pest"),
    ("pest",               "pest"),
    ("disease",            "pest"),
    ("insect",             "pest"),
    ("bug",                "pest"),
    ("fungus",             "pest"),
    ("blight",             "pest"),
    ("wilt",               "pest"),
    ("yellowing",          "pest"),
    ("leaf curl",          "pest"),
    ("stem borer",         "pest"),
    ("girdle beetle",      "pest"),
    ("yellow mosaic",      "pest"),
    ("spray",              "pest"),
    # ── Market / MSP ──────────────────────────────────────────────────────
    ("msp",                "market"),
    ("market price",       "market"),
    ("mandi",              "market"),
    ("bhav",               "market"),
    ("price",              "market"),
    ("rate",               "market"),
    # ── Schemes ───────────────────────────────────────────────────────────
    ("scheme",             "scheme"),
    ("pm-kisan",           "scheme"),
    ("pmfby",              "scheme"),
    ("insurance",          "scheme"),
    ("subsidy",            "scheme"),
    ("yojana",             "scheme"),
    ("sarkar",             "scheme"),
    # ── Weather ───────────────────────────────────────────────────────────
    ("weather",            "weather"),
    ("rain",               "weather"),
    ("temperatur",         "weather"),
    ("heat",               "weather"),
    ("forecast",           "weather"),
    ("barish",             "weather"),
    ("garmi",              "weather"),    # Hindi: heat
]


# ── Stage / body-part context words that enrich follow-up messages ────────────
STAGE_WORDS    = {"leaf", "leaves", "stem", "root", "flower", "boll", "fruit",
                  "pod", "grain", "seedling", "tiller", "branch", "shoot"}
GROWTH_STAGES  = {"vegetative", "flowering", "germination", "sowing", "harvesting",
                  "maturity", "transplant", "nursery", "tillering", "jointing"}


def _detect_categories(msg: str) -> list[str]:
    """Return ALL matched categories for a message, in priority order (no duplicates)."""
    seen = []
    for keyword, category in KEYWORD_CATEGORY_MAP:
        if keyword in msg and category not in seen:
            seen.append(category)
    return seen


def _enrich_from_history(conversation_history: list) -> dict:
    """
    Scan the last few turns of conversation history to recover:
    - topic (last agent-response category)
    - crop mentioned by the farmer
    """
    ctx = {"topic": None, "crop": None}
    for turn in reversed(conversation_history[-8:]):
        content = turn.get("content", "").lower()
        if turn.get("role") == "user":
            # Try to find a crop name mentioned
            for alias, canonical in CROP_ALIASES.items():
                if alias in content:
                    ctx["crop"] = canonical
                    break
            if not ctx["crop"]:
                for crop in ["wheat", "cotton", "rice", "soybean", "sugarcane",
                             "onion", "chickpea", "tur dal", "maize", "groundnut"]:
                    if crop in content:
                        ctx["crop"] = crop
                        break
    return ctx


# Fallback generic responses when no keyword matches at all
DEMO_FALLBACKS = [
    "🌾 **I can help! Here are the most popular topics farmers ask me:**\n\n| Ask me about | Example question |\n|---|---|\n| 🌱 Crop recommendation | \"Which crop for kharif in Vidarbha?\" |\n| 🌿 Sowing guide | \"When should I sow soybean?\" |\n| 💊 Fertilizer | \"Fertilizer schedule for cotton\" |\n| 🌿 Weed control | \"Which herbicide for soybean at 20 DAS?\" |\n| 🐛 Pest & disease | \"Pests in soybean vegetative stage\" |\n| 💧 Irrigation | \"Irrigation plan for wheat\" |\n| 📊 MSP prices | \"MSP for soybean 2024\" |\n| 🏛️ Govt schemes | \"PM-KISAN eligibility\" |\n\nOr click a **Quick Advisory** button on the left panel. 🌾",
    "🌱 **Vidarbha Soil Guide**\n\nBlack cotton soil (Vertisols) dominates Nagpur–Yavatmal–Amravati region:\n- 🟫 Deep, dark, high clay content\n- 💧 Excellent water holding — avoid waterlogging\n- 🌿 Low in N and Zn — apply Urea + ZnSO₄\n- 🌾 **Best crops:** Soybean, Cotton, Tur Dal, Wheat (rabi)\n\nGet a **Soil Health Card** free from your nearest KVK or Agriculture office. 🌾",
    "📱 **Farmer Resources — Nagpur, Vidarbha**\n\n| Resource | Contact |\n|---|---|\n| KVK Nagpur | 0712-2500849 |\n| Kisan Call Centre | 1800-180-1551 (Free 24×7) |\n| APMC Mandi Prices | agmarknet.gov.in |\n| Soil Health Card | soilhealth.dac.gov.in |\n| VNMKV (Agriculture Uni.) | vnmkv.ac.in |\n| IMD Weather | imd.gov.in |\n\n*Type your question in any language — English, हिंदी, or मराठी!* 🌾",
]
_fallback_idx = 0


def get_demo_response(
    user_message: str,
    farm_context: dict | None = None,
    conversation_history: list | None = None,
) -> str:
    """
    Context-aware demo response engine.

    Priority:
    1. Detect ALL categories in the current message.
    2. For each category, resolve the crop (message alias → session ctx → history).
    3. If the message is a short follow-up (leaf / stage words), inherit last topic
       from conversation history and generate a combined response.
    4. If multiple categories detected, generate a combined answer.
    5. Fall back to rotating generic tips.
    """
    global _fallback_idx

    msg  = user_message.lower().strip()
    ctx  = farm_context or {}

    # ── 1. Resolve crop ──────────────────────────────────────────────────────
    # a) Try alias match inside the current message
    crop = ""
    for alias, canonical in CROP_ALIASES.items():
        if alias in msg:
            crop = canonical
            break
    # b) Try plain crop name in message
    if not crop:
        for c in ["wheat", "cotton", "rice", "soybean", "sugarcane",
                  "onion", "chickpea", "tur dal", "maize", "groundnut"]:
            if c in msg:
                crop = c
                break
    # c) Fall back to session farm context
    if not crop:
        crop = normalise_crop(ctx.get("crop", ""))
    # d) Fall back to conversation history
    if not crop and conversation_history:
        history_ctx = _enrich_from_history(conversation_history)
        crop = history_ctx.get("crop") or ""

    # ── 2. Detect categories in current message ──────────────────────────────
    categories = _detect_categories(msg)

    # ── 3. Short follow-up detection ─────────────────────────────────────────
    #   e.g. "leaf", "vegetative, pest", "stem borer" — very short messages
    #   where the farmer continues a previous thread.
    words = set(msg.replace(",", " ").split())
    is_short = len(words) <= 5
    has_stage  = bool(words & STAGE_WORDS)
    has_growth = bool(words & GROWTH_STAGES)

    # If short message with stage/body-part words and no strong category,
    # treat it as a pest/disease follow-up (most common scenario)
    if is_short and (has_stage or has_growth) and not categories:
        categories = ["pest"]

    # If message explicitly mentions vegetative + pest (any combo), prioritise pest
    if ("vegetative" in msg or "growth stage" in msg) and (
        "pest" in msg or "disease" in msg or "insect" in msg
    ):
        if "pest" not in categories:
            categories.insert(0, "pest")

    # ── 4. Build combined response for multi-category messages ───────────────
    #   e.g. "when should I sow soybean and which pesticide to use"
    #   → gives both sowing guide AND pest guide
    responses = []
    for cat in categories[:2]:   # max 2 topics per reply to keep it readable
        cat_responses = DEMO_RESPONSES_KEYED.get(cat, {})

        if cat == "crop_rec":
            # crop_rec uses season as key, not crop name
            season = ctx.get("season", "")
            if "rabi" in msg or "rabi" in season:
                season_key = "rabi"
            elif "kharif" in msg or "kharif" in season:
                season_key = "kharif"
            else:
                season_key = "default"
            resp = cat_responses.get(season_key) or cat_responses.get("default", "")
            if resp:
                responses.append(resp)
        else:
            for crop_key in [crop, "default"]:
                if crop_key in cat_responses:
                    responses.append(cat_responses[crop_key])
                    break

    if responses:
        combined = "\n\n---\n\n".join(responses)
        return (
            f"> 🟡 **Demo Mode** — Add your IBM Watsonx API key to `.env` for live AI responses.\n\n"
            f"{combined}"
        )

    # ── 5. Generic fallback (rotating) ───────────────────────────────────────
    resp = DEMO_FALLBACKS[_fallback_idx % len(DEMO_FALLBACKS)]
    _fallback_idx += 1
    return (
        f"> 🟡 **Demo Mode** — Add your IBM Watsonx API key to `.env` for live AI responses.\n\n"
        f"{resp}"
    )


# ── LLM call ──────────────────────────────────────────────────────────────────
def query_watsonx(user_message: str, conversation_history: list, farm_context: dict) -> str:
    model = get_watsonx_model()
    if model is None:
        return get_demo_response(user_message, farm_context, conversation_history)

    system_prompt = build_system_prompt(farm_context)

    # Build conversation string (Granite instruct format)
    history_text = ""
    for turn in conversation_history[-6:]:          # last 3 exchanges
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            history_text += f"\nFarmer: {content}"
        else:
            history_text += f"\nAgent: {content}"

    prompt = (
        f"{system_prompt}\n"
        f"--- Conversation ---{history_text}\n"
        f"Farmer: {user_message}\n"
        f"Agent:"
    )

    try:
        result   = model.generate_text(prompt=prompt)
        response = result.strip() if isinstance(result, str) else str(result)
        return response
    except Exception as exc:
        logger.error(f"Watsonx generation error: {exc}")
        return f"⚠️ I'm having trouble connecting to the AI service right now. Please try again in a moment.\n\nError: {exc}"


# ── Image processing (pest/disease) ──────────────────────────────────────────
def process_farm_image(image_data: str) -> str:
    """Convert base64 image to description for text-based Granite model."""
    try:
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img       = Image.open(BytesIO(img_bytes))
        width, height = img.size
        mode = img.mode
        return (
            f"[Farmer uploaded a {width}x{height} {mode} image. "
            "Please analyze this as a potential crop pest/disease image. "
            "Ask clarifying questions about: crop type, affected plant part, "
            "duration of symptoms, and approximate field area affected.]"
        )
    except Exception as exc:
        logger.warning(f"Image processing error: {exc}")
        return "[Farmer uploaded an image for pest/disease identification.]"


# =============================================================================
#  ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the main application page."""
    if "conversation" not in session:
        session["conversation"] = []
    if "farm_context" not in session:
        session["farm_context"] = {}
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint."""
    data        = request.get_json(silent=True) or {}
    user_msg    = data.get("message", "").strip()
    image_data  = data.get("image")
    farm_ctx    = data.get("farm_context", {})

    if not user_msg and not image_data:
        return jsonify({"error": "Empty message"}), 400

    # Merge farm context into session
    if farm_ctx:
        session["farm_context"] = {**session.get("farm_context", {}), **farm_ctx}
        session.modified = True

    # Append image description to message if present
    if image_data:
        img_desc  = process_farm_image(image_data)
        user_msg  = f"{user_msg}\n{img_desc}" if user_msg else img_desc

    conversation = session.get("conversation", [])
    conversation.append({"role": "user", "content": user_msg})

    response = query_watsonx(user_msg, conversation, session.get("farm_context", {}))

    conversation.append({"role": "assistant", "content": response})
    session["conversation"] = conversation[-20:]    # keep last 20 turns
    session.modified = True

    return jsonify({
        "response":  response,
        "mode":      "live" if get_watsonx_model() else "demo",
        "turn_count": len(conversation) // 2,
    })


@app.route("/api/update-farm-context", methods=["POST"])
def update_farm_context():
    """Save farm profile / soil / weather data to session."""
    data = request.get_json(silent=True) or {}
    session["farm_context"] = {**session.get("farm_context", {}), **data}
    session.modified = True
    return jsonify({"status": "ok", "context": session["farm_context"]})


@app.route("/api/clear-chat", methods=["POST"])
def clear_chat():
    """Reset conversation history."""
    session["conversation"] = []
    session.modified = True
    return jsonify({"status": "cleared"})


@app.route("/api/quick-advice", methods=["POST"])
def quick_advice():
    """One-shot advisory without conversation history."""
    data     = request.get_json(silent=True) or {}
    category = data.get("category", "general")   # fertilizer | irrigation | pest | weather | market
    crop     = data.get("crop", "")
    context  = data.get("context", {})

    prompts = {
        "fertilizer":  f"Give a precise fertilizer schedule for {crop} crop for the upcoming season in {REGION}.",
        "irrigation":  f"Provide irrigation schedule and water requirement for {crop} crop.",
        "pest":        f"List the top 5 pests and diseases affecting {crop} in {REGION} with organic management tips.",
        "weather":     f"Give weather-based farming tips for {crop} during extreme heat (>40°C) in {REGION}.",
        "market":      f"Provide current MSP and market price awareness for {crop} in India.",
        "schemes":     f"List 5 relevant government agricultural schemes for {crop} farmers in India with application details.",
        "general":     f"Give top 5 expert farming tips for {crop} cultivation in {REGION}.",
    }

    # Inject crop into context for demo keyword matching
    context_with_crop = {**context, "crop": crop} if crop else context
    message  = prompts.get(category, prompts["general"])
    response = query_watsonx(message, [], context_with_crop)
    return jsonify({"response": response, "category": category, "crop": crop})


@app.route("/api/status", methods=["GET"])
def status():
    """Health check and configuration status."""
    model_ready = get_watsonx_model() is not None
    return jsonify({
        "status":       "running",
        "mode":         "live" if model_ready else "demo",
        "model":        MODEL_ID,
        "region":       REGION,
        "language":     LANGUAGE_SUPPORT,
        "tone":         AGENT_TONE,
        "crops":        CROP_SPECIALIZATIONS,
        "watsonx_available": WATSONX_AVAILABLE,
    })


@app.route("/api/setup-check", methods=["GET"])
def setup_check():
    """Returns a detailed checklist including the exact connection error if any."""
    api_key    = os.getenv("IBM_API_KEY", "").strip()
    project_id = os.getenv("IBM_PROJECT_ID", "").strip()
    wx_url     = os.getenv("IBM_WATSONX_URL", "").strip()

    model_ok = get_watsonx_model() is not None

    checks = {
        "env_file_loaded":  bool(api_key or project_id),
        "api_key_set":      bool(api_key and api_key != "your_ibm_cloud_api_key_here"),
        "project_id_set":   bool(project_id and project_id != "your_watsonx_project_id_here"),
        "sdk_installed":    WATSONX_AVAILABLE,
        "model_connected":  model_ok,
    }
    all_ok  = all(checks.values())
    missing = [k for k, v in checks.items() if not v]

    # Expose the human-readable error for the UI
    error_detail = _watsonx_error if not model_ok else None

    return jsonify({
        "checks":       checks,
        "all_ok":       all_ok,
        "missing":      missing,
        "mode":         "live" if model_ok else "demo",
        "model_used":   _watsonx_model_used,
        "watsonx_url":  wx_url or "https://us-south.ml.cloud.ibm.com",
        "error_detail": error_detail,
        "next_step":    _next_step_message(missing, error_detail),
    })


def _next_step_message(missing: list, error_detail: str | None = None) -> str:
    if not missing:
        return f"✅ Connected! Model: {_watsonx_model_used or MODEL_ID}"
    if "sdk_installed" in missing:
        return "📦 Run in your terminal: pip install ibm-watsonx-ai"
    if "api_key_set" in missing:
        return "🔑 Set IBM_API_KEY in your .env file (IBM Cloud → Manage → IAM → API keys)"
    if "project_id_set" in missing:
        return "📋 Set IBM_PROJECT_ID in your .env file (Watsonx Studio → Project → Manage → General)"
    if "model_connected" in missing:
        if error_detail:
            return f"❌ {error_detail}"
        return "🔄 Credentials found but connection failed — click 'Test Connection' below for details."
    return "⚙️ Check your .env configuration and restart Flask."


@app.route("/api/diagnose", methods=["GET"])
def diagnose():
    """
    Run a live connection test and return detailed diagnostics.
    Always re-tests — does NOT use the cached model.
    """
    api_key    = os.getenv("IBM_API_KEY", "").strip()
    project_id = os.getenv("IBM_PROJECT_ID", "").strip()
    url        = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()

    result = {
        "sdk_available":  WATSONX_AVAILABLE,
        "api_key_prefix": (api_key[:6] + "…") if len(api_key) > 6 else "(not set)",
        "project_id_prefix": (project_id[:8] + "…") if len(project_id) > 8 else "(not set)",
        "watsonx_url":    url,
        "models_tried":   [],
        "connected":      False,
        "model_used":     None,
        "error":          None,
        "raw_error":      None,
    }

    if not WATSONX_AVAILABLE:
        result["error"] = "ibm-watsonx-ai SDK not installed. Run: pip install ibm-watsonx-ai"
        return jsonify(result)

    if not api_key or api_key == "your_ibm_cloud_api_key_here":
        result["error"] = "IBM_API_KEY not set in .env"
        return jsonify(result)

    if not project_id or project_id == "your_watsonx_project_id_here":
        result["error"] = "IBM_PROJECT_ID not set in .env"
        return jsonify(result)

    # Step 1: Authenticate
    try:
        credentials = Credentials(api_key=api_key, url=url)
        client      = APIClient(credentials)
    except Exception as exc:
        result["error"]     = f"Authentication failed: {type(exc).__name__}: {exc}"
        result["raw_error"] = str(exc)
        return jsonify(result)

    # Step 2: Try each model
    for mid in FALLBACK_MODEL_IDS:
        entry = {"model_id": mid, "status": "trying", "error": None}
        try:
            m = ModelInference(
                model_id   = mid,
                api_client = client,
                project_id = project_id,
                params     = {GenParams.MAX_NEW_TOKENS: 20, GenParams.TEMPERATURE: 0.1},
            )
            test_out = m.generate_text(prompt="Say: OK")
            entry["status"] = "success"
            entry["test_output"] = str(test_out)[:80]
            result["models_tried"].append(entry)
            result["connected"]  = True
            result["model_used"] = mid
            result["error"]      = None

            # Also refresh the cached model
            reset_watsonx_model()
            get_watsonx_model()
            break
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"]  = f"{type(exc).__name__}: {str(exc)[:200]}"
            result["models_tried"].append(entry)
            result["raw_error"] = str(exc)
            continue

    if not result["connected"]:
        result["error"] = _format_watsonx_error(
            Exception(result.get("raw_error", "All models failed"))
        )

    return jsonify(result)


@app.route("/api/retry-connection", methods=["POST"])
def retry_connection():
    """Force-reset the cached model and re-attempt connection."""
    reset_watsonx_model()
    model = get_watsonx_model()
    return jsonify({
        "connected":  model is not None,
        "model_used": _watsonx_model_used,
        "error":      _watsonx_error,
        "mode":       "live" if model else "demo",
    })


# =============================================================================
if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
