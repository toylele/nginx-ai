from pathlib import Path
import sys

# ensure project root is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cleaner.enricher import GeoIPEnricher


def test_geoip_enricher_does_not_raise_and_adds_fields():
    enricher = GeoIPEnricher(ROOT)
    rec = {'remote_addr': '8.8.8.8'}
    enriched = enricher.enrich(rec.copy())

    # Fields should exist (may be None if GeoIP DB not present)
    assert 'geo_country' in enriched
    assert 'geo_city' in enriched
    assert 'geo_latitude' in enriched
    assert 'geo_longitude' in enriched

    # Values should be either None or of expected types
    assert (enriched['geo_country'] is None) or isinstance(enriched['geo_country'], str)
    if enriched['geo_latitude'] is not None:
        assert isinstance(enriched['geo_latitude'], float)
    if enriched['geo_longitude'] is not None:
        assert isinstance(enriched['geo_longitude'], float)
