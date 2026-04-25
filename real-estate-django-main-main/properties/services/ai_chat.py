import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a strict real estate filter extractor.

Return ONLY valid JSON with:
- "filters"
- "message"

RULES:
1. "under X", "below X", "max X" → max_price = X
2. "over X", "above X", "at least X" → min_price = X
3. "apartments" / "apartment" → property_type = "apartment"
   "houses" / "house" → property_type = "house"
   "villas" / "villa" → property_type = "villa"
   "studios" / "studio" → property_type = "studio"
   "land" / "lands" → property_type = "land"
   "commercial" → property_type = "commercial"
4. "for sale" / "to buy" → listing_type = "sale"
5. "for rent" / "to rent" → listing_type = "rent"
6. ACCUMULATE filters across the conversation — keep previously set filters unless the user explicitly changes or removes one.
7. Never guess or invent values not mentioned by the user.
8. If the user removes a constraint (e.g. "any price", "any city"), set that filter to null.

FILTER SCHEMA:
{
  "city": string or null,
  "listing_type": "sale" | "rent" | null,
  "property_type": "apartment" | "studio" | "house" | "villa" | "land" | "commercial" | null,
  "min_price": number or null,
  "max_price": number or null,
  "min_area": number or null,
  "bedrooms": number or null,
  "bathrooms": number or null
}

MESSAGE RULES:
- Write 1 short sentence summarising what you searched for.
- Mention the active filters (city, type, price, bedrooms) naturally.
- Examples:
  "Here are houses in Chicago under $300,000."
  "Showing apartments for rent in Los Angeles."
  "Found villas with at least 4 bedrooms."
- Never say "I found X results" — you don't know the count.
- Never ask a follow-up question.

Return ONLY JSON. No markdown.
"""


def build_prompt(chat_history):
    conversation = "\n".join(
        f"{msg.get('role','user').upper()}: {msg.get('content','')}"
        for msg in chat_history
    )
    return f"{SYSTEM_PROMPT}\n\nConversation:\n{conversation}\n\nJSON:"


def call_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 2048},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()



def extract_price(text):
    if text is None:
        return None

    text = str(text).lower().replace(",", "").strip()

    match = re.search(r"(\d+)\s*k", text)
    if match:
        return int(match.group(1)) * 1000

    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))

    return None


def safe_int(value):
    try:
        if value is None:
            return None
        return int(str(value).replace(",", "").strip())
    except:
        return None



_PROPERTY_TYPE_KEYWORDS = {
    "apartments": "apartment", "apartment": "apartment",
    "houses": "house", "house": "house",
    "villas": "villa", "villa": "villa",
    "studios": "studio", "studio": "studio",
    "commercial": "commercial",
    "land": "land",
}

_LISTING_TYPE_KEYWORDS = {
    "for rent": "rent", "to rent": "rent",
    "for sale": "sale", "to buy": "sale", "to purchase": "sale",
}


def apply_keyword_overrides(filters, user_message):
    msg = user_message.lower()
    for keyword, ptype in _PROPERTY_TYPE_KEYWORDS.items():
        if re.search(r'\b' + keyword + r'\b', msg):
            filters["property_type"] = ptype
            break
    for phrase, ltype in _LISTING_TYPE_KEYWORDS.items():
        if phrase in msg:
            filters["listing_type"] = ltype
            break
    return filters


def normalize_filters(filters):
    if not isinstance(filters, dict):
        return {}

    for key in ["max_price", "min_price"]:
        value = filters.get(key)

        if isinstance(value, str):
            filters[key] = extract_price(value)
        else:
            filters[key] = safe_int(value)

    return filters



ALLOWED_PROPERTY_TYPES = {
    "apartment", "studio", "house", "villa", "land", "commercial"
}

ALLOWED_LISTING_TYPES = {"sale", "rent"}


def validate_filters(filters):
    if not isinstance(filters, dict):
        return {}

    if filters.get("property_type") not in ALLOWED_PROPERTY_TYPES:
        filters["property_type"] = None

    if filters.get("listing_type") not in ALLOWED_LISTING_TYPES:
        filters["listing_type"] = None

    filters["max_price"] = safe_int(filters.get("max_price"))
    filters["min_price"] = safe_int(filters.get("min_price"))
    filters["bedrooms"] = safe_int(filters.get("bedrooms"))
    filters["bathrooms"] = safe_int(filters.get("bathrooms"))

    return filters


def build_message_from_filters(filters):
    parts = []

    ptype = filters.get("property_type")
    ltype = filters.get("listing_type")
    city  = filters.get("city")
    max_p = filters.get("max_price")
    min_p = filters.get("min_price")
    beds  = filters.get("bedrooms")

    type_label = (ptype.capitalize() + "s") if ptype else "Properties"
    parts.append(type_label)

    if ltype == "rent":
        parts.append("for rent")
    elif ltype == "sale":
        parts.append("for sale")

    if city:
        parts.append(f"in {city}")

    if max_p is not None:
        parts.append(f"under ${max_p:,}")
    elif min_p is not None:
        parts.append(f"over ${min_p:,}")

    if beds is not None:
        parts.append(f"with {beds}+ bedrooms")

    return " ".join(parts) + "."


def parse_ollama_response(raw):
    text = raw.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError("No JSON found")

    parsed = json.loads(text[start:end])

    if "filters" not in parsed:
        raise ValueError("Missing filters")

    filters = parsed["filters"]
    filters = normalize_filters(filters)
    filters = validate_filters(filters)

    return {
        "filters": filters,
        "message": build_message_from_filters(filters)
    }



def apply_filters(queryset, filters):
    filters = normalize_filters(filters)
    filters = validate_filters(filters)

    if filters.get("city"):
        queryset = queryset.filter(city__icontains=filters["city"])

    if filters.get("listing_type"):
        queryset = queryset.filter(listing_type=filters["listing_type"])

    if filters.get("property_type"):
        queryset = queryset.filter(property_type=filters["property_type"])

    if filters.get("max_price") is not None:
        queryset = queryset.filter(price__lte=filters["max_price"])

    if filters.get("min_price") is not None:
        queryset = queryset.filter(price__gte=filters["min_price"])

    if filters.get("bedrooms") is not None:
        queryset = queryset.filter(bedrooms__gte=filters["bedrooms"])

    if filters.get("bathrooms") is not None:
        queryset = queryset.filter(bathrooms__gte=filters["bathrooms"])

    return queryset



COMPARISON_SYSTEM_PROMPT = """You are a helpful real estate assistant.
Compare the properties using exactly this format:

**[Property 1 name] vs [Property 2 name]**

• **Price:** $X vs $Y
• **Price/m²:** $X vs $Y
• **Size:** Xm² vs Ym²
• **Beds/Baths:** X bed X bath vs Y bed Y bath
• **Location:** City1 vs City2

**Verdict:** One sentence saying which is the better deal and why.

Only use the provided data. No extra lines or commentary."""


def detect_intent(message, queryset):
    """Return ("compare", [prop, ...]) or ("filter", [])."""
    found = []
    seen_ids = set()

    id_matches = re.findall(
        r'(?:#|(?:property|id|listing)\s+)(\d+)', message, re.IGNORECASE
    )
    for id_str in id_matches:
        try:
            prop = queryset.filter(id=int(id_str)).first()
            if prop and prop.id not in seen_ids:
                found.append(prop)
                seen_ids.add(prop.id)
        except (ValueError, TypeError):
            pass

    if len(found) < 2:
        for prop in queryset:
            if prop.id in seen_ids or len(prop.name) < 4:
                continue
            if re.search(r'\b' + re.escape(prop.name) + r'\b', message, re.IGNORECASE):
                found.append(prop)
                seen_ids.add(prop.id)
            if len(found) >= 4:
                break

    if len(found) >= 2:
        return ("compare", found[:4])
    return ("filter", [])


def serialize_property_for_comparison(prop):
    features = list(prop.features.values_list('name', flat=True))
    price_per_m2 = (
        round(float(prop.price) / float(prop.area), 2) if prop.area else None
    )
    return {
        "id": prop.id,
        "name": prop.name,
        "city": prop.city,
        "location": prop.location,
        "price": float(prop.price),
        "area": float(prop.area),
        "price_per_m2": price_per_m2,
        "property_type": prop.get_property_type_display(),
        "listing_type": prop.get_listing_type_display(),
        "bedrooms": prop.bedrooms,
        "bathrooms": prop.bathrooms,
        "rooms": prop.rooms,
        "features": features,
        "custom_features": prop.custom_features or "",
    }


def build_comparison_prompt(props_data, chat_history):
    props_text = "\n\n".join(
        "Property {n}: {name} (ID: {id})\n"
        "  City: {city}, Location: {location}\n"
        "  Price: ${price:,.0f} ({listing_type})\n"
        "  Price per m²: ${price_per_m2}\n"
        "  Type: {property_type}, Area: {area} m²\n"
        "  Bedrooms: {bedrooms}, Bathrooms: {bathrooms}\n"
        "  Features: {features}\n"
        "  Additional: {custom_features}".format(
            n=i + 1,
            features=", ".join(p["features"]) or "None",
            **{k: v for k, v in p.items() if k != "features"},
        )
        for i, p in enumerate(props_data)
    )
    last_user_msg = next(
        (m["content"] for m in reversed(chat_history) if m["role"] == "user"),
        "Which is better?",
    )
    return (
        f"{COMPARISON_SYSTEM_PROMPT}\n\n"
        f"Properties:\n{props_text}\n\n"
        f"User question: {last_user_msg}\n\nYour comparison:"
    )