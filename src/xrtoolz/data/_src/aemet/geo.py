"""Spanish administrative geocoding helpers for AEMET stations.

AEMET's inventory endpoint returns only ``provincia`` (plus sometimes
inconsistent abbreviations for the same province). The autonomous
community is never included in the raw payload. This module adds:

- :func:`canonical_province` — collapses AEMET's raw spellings
  (``STA. CRUZ DE TENERIFE``, ``SANTA CRUZ DE TENERIFE``,
  ``BALEARES``, ``ILLES BALEARS``) to one canonical name per province.
- :data:`PROVINCE_TO_COMMUNITY` — the public 50-province →
  17-community (+ 2 autonomous cities) mapping for Spain.
- :func:`community_for` — look up the community for a raw AEMET
  ``provincia`` string.

Names follow standard Spanish usage (``Cataluña``, ``País Vasco``) so
cross-dataset joins are predictable.
"""

from __future__ import annotations

import unicodedata


def _norm(text: str) -> str:
    """Uppercase, strip, drop accents — what AEMET does inconsistently."""
    stripped = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    return " ".join(stripped.upper().split())


# Raw AEMET ``provincia`` variants → canonical province label.
# Keys are normalised with ``_norm`` (no accents, uppercased, collapsed
# whitespace). Values use standard Spanish spelling with accents.
_CANONICAL_PROVINCE: dict[str, str] = {
    # Galicia
    "A CORUNA": "A Coruña",
    "LA CORUNA": "A Coruña",
    "LUGO": "Lugo",
    "OURENSE": "Ourense",
    "PONTEVEDRA": "Pontevedra",
    # Principado de Asturias
    "ASTURIAS": "Asturias",
    # Cantabria
    "CANTABRIA": "Cantabria",
    # País Vasco
    "ARABA/ALAVA": "Álava",
    "ALAVA": "Álava",
    "BIZKAIA": "Bizkaia",
    "VIZCAYA": "Bizkaia",
    "GIPUZKOA": "Gipuzkoa",
    "GUIPUZCOA": "Gipuzkoa",
    # Navarra
    "NAVARRA": "Navarra",
    # La Rioja
    "LA RIOJA": "La Rioja",
    # Aragón
    "HUESCA": "Huesca",
    "TERUEL": "Teruel",
    "ZARAGOZA": "Zaragoza",
    # Cataluña
    "BARCELONA": "Barcelona",
    "GIRONA": "Girona",
    "GERONA": "Girona",
    "LLEIDA": "Lleida",
    "LERIDA": "Lleida",
    "TARRAGONA": "Tarragona",
    # Comunidad Valenciana
    "ALICANTE": "Alicante",
    "CASTELLON": "Castellón",
    "VALENCIA": "Valencia",
    # Región de Murcia
    "MURCIA": "Murcia",
    # Andalucía
    "ALMERIA": "Almería",
    "CADIZ": "Cádiz",
    "CORDOBA": "Córdoba",
    "GRANADA": "Granada",
    "HUELVA": "Huelva",
    "JAEN": "Jaén",
    "MALAGA": "Málaga",
    "SEVILLA": "Sevilla",
    # Extremadura
    "BADAJOZ": "Badajoz",
    "CACERES": "Cáceres",
    # Castilla-La Mancha
    "ALBACETE": "Albacete",
    "CIUDAD REAL": "Ciudad Real",
    "CUENCA": "Cuenca",
    "GUADALAJARA": "Guadalajara",
    "TOLEDO": "Toledo",
    # Castilla y León
    "AVILA": "Ávila",
    "BURGOS": "Burgos",
    "LEON": "León",
    "PALENCIA": "Palencia",
    "SALAMANCA": "Salamanca",
    "SEGOVIA": "Segovia",
    "SORIA": "Soria",
    "VALLADOLID": "Valladolid",
    "ZAMORA": "Zamora",
    # Comunidad de Madrid
    "MADRID": "Madrid",
    # Illes Balears
    "BALEARES": "Illes Balears",
    "ILLES BALEARS": "Illes Balears",
    # Canarias
    "LAS PALMAS": "Las Palmas",
    "SANTA CRUZ DE TENERIFE": "Santa Cruz de Tenerife",
    "STA. CRUZ DE TENERIFE": "Santa Cruz de Tenerife",
    "STA CRUZ DE TENERIFE": "Santa Cruz de Tenerife",
    # Autonomous cities
    "CEUTA": "Ceuta",
    "MELILLA": "Melilla",
}


# Canonical province → autonomous community (Spanish names).
PROVINCE_TO_COMMUNITY: dict[str, str] = {
    # Galicia
    "A Coruña": "Galicia",
    "Lugo": "Galicia",
    "Ourense": "Galicia",
    "Pontevedra": "Galicia",
    # Single-province communities
    "Asturias": "Principado de Asturias",
    "Cantabria": "Cantabria",
    "Navarra": "Comunidad Foral de Navarra",
    "La Rioja": "La Rioja",
    "Madrid": "Comunidad de Madrid",
    "Murcia": "Región de Murcia",
    "Illes Balears": "Illes Balears",
    # País Vasco
    "Álava": "País Vasco",
    "Bizkaia": "País Vasco",
    "Gipuzkoa": "País Vasco",
    # Aragón
    "Huesca": "Aragón",
    "Teruel": "Aragón",
    "Zaragoza": "Aragón",
    # Cataluña
    "Barcelona": "Cataluña",
    "Girona": "Cataluña",
    "Lleida": "Cataluña",
    "Tarragona": "Cataluña",
    # Comunidad Valenciana
    "Alicante": "Comunidad Valenciana",
    "Castellón": "Comunidad Valenciana",
    "Valencia": "Comunidad Valenciana",
    # Andalucía
    "Almería": "Andalucía",
    "Cádiz": "Andalucía",
    "Córdoba": "Andalucía",
    "Granada": "Andalucía",
    "Huelva": "Andalucía",
    "Jaén": "Andalucía",
    "Málaga": "Andalucía",
    "Sevilla": "Andalucía",
    # Extremadura
    "Badajoz": "Extremadura",
    "Cáceres": "Extremadura",
    # Castilla-La Mancha
    "Albacete": "Castilla-La Mancha",
    "Ciudad Real": "Castilla-La Mancha",
    "Cuenca": "Castilla-La Mancha",
    "Guadalajara": "Castilla-La Mancha",
    "Toledo": "Castilla-La Mancha",
    # Castilla y León
    "Ávila": "Castilla y León",
    "Burgos": "Castilla y León",
    "León": "Castilla y León",
    "Palencia": "Castilla y León",
    "Salamanca": "Castilla y León",
    "Segovia": "Castilla y León",
    "Soria": "Castilla y León",
    "Valladolid": "Castilla y León",
    "Zamora": "Castilla y León",
    # Canarias
    "Las Palmas": "Canarias",
    "Santa Cruz de Tenerife": "Canarias",
    # Autonomous cities
    "Ceuta": "Ceuta",
    "Melilla": "Melilla",
}


def canonical_province(raw: str | None) -> str | None:
    """Return the canonical province name for an AEMET ``provincia`` value.

    Returns ``None`` if ``raw`` is empty or unrecognised. Unknown values
    degrade gracefully so a new AEMET province spelling doesn't crash
    the inventory parser.
    """
    if not raw:
        return None
    key = _norm(raw)
    return _CANONICAL_PROVINCE.get(key)


def community_for(raw_province: str | None) -> str | None:
    """Return the autonomous community for an AEMET ``provincia`` value."""
    prov = canonical_province(raw_province)
    if prov is None:
        return None
    return PROVINCE_TO_COMMUNITY.get(prov)
