"""
CloudMorph Provider Pricing Service.

Provider adapters:
- Azure: public Retail Prices API (no cloud credentials required).
- AWS: public AWS Price List Bulk API files (no AWS account credentials required),
  with a six-hour in-memory regional offer-file cache.
- GCP: public InstanceFinder daily dataset (no credentials or API key).

Prices are public On-Demand/Consumption list prices, not customer invoices.
Estimates cover VM compute only; storage, networking, taxes, discounts,
licenses and other services may add cost.
"""

import asyncio
import copy
import csv
import gzip
import json
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ENGINE_NAME = "CloudMorph Provider Pricing Service"
ENGINE_VERSION = "1.5.0"

AZURE_RETAIL_PRICES_ENDPOINT = "https://prices.azure.com/api/retail/prices"
AZURE_RETAIL_PRICES_SOURCE = (
    "https://learn.microsoft.com/en-us/rest/api/cost-management/"
    "retail-prices/azure-retail-prices"
)
AWS_PRICING_SOURCE = (
    "https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/"
    "finding-prices-in-service-price-list-files.html"
)
AWS_PRICE_LIST_BASE = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current"
GCP_PUBLIC_DATA_BASE = "https://instancefinder.com/data/gcp"
GCP_PUBLIC_DATA_SOURCE = "https://instancefinder.com/dataset/"
GCP_PUBLIC_DATA_CACHE_TTL_SECONDS = 6 * 60 * 60

# Cache exact VM price results so repeat comparisons do not call public APIs
# for every candidate again. Successful prices live for 15 minutes;
# unavailable/ambiguous results live for 2 minutes to avoid stale failures.
VM_PRICE_CACHE_TTL_SECONDS = 15 * 60
VM_PRICE_FAILURE_CACHE_TTL_SECONDS = 2 * 60
_VM_PRICE_RESULT_CACHE = {}
_VM_PRICE_IN_FLIGHT = {}
_VM_PRICE_CACHE_LOCK = asyncio.Lock()
_GCP_PUBLIC_DATA_CACHE = {}
_GCP_PUBLIC_DATA_CACHE_LOCK = threading.Lock()

HOURS_PER_ESTIMATED_MONTH = 730
DEFAULT_TIMEOUT_SECONDS = 20
MAX_PAGES = 5

# AWS regional EC2 CSV is still large, but considerably smaller than JSON.
# Stream it to disk once, then scan it without loading the full file into RAM.
AWS_PRICE_CACHE_TTL_SECONDS = 6 * 60 * 60
AWS_PRICE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "cloudmorph_aws_price_cache")
_AWS_PRICE_REGION_LOCKS = {}
_AWS_PRICE_CACHE_LOCK = threading.Lock()

SUPPORTED_CURRENCIES = {
    "USD", "INR", "EUR", "GBP", "AUD", "CAD", "SGD", "JPY"
}

# AWS Price List API expects the human-readable AWS location name.
AWS_REGION_LOCATIONS = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "af-south-1": "Africa (Cape Town)",
    "ap-east-1": "Asia Pacific (Hong Kong)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-south-2": "Asia Pacific (Hyderabad)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-southeast-3": "Asia Pacific (Jakarta)",
    "ap-southeast-4": "Asia Pacific (Melbourne)",
    "ca-central-1": "Canada (Central)",
    "ca-west-1": "Canada West (Calgary)",
    "eu-central-1": "EU (Frankfurt)",
    "eu-central-2": "Europe (Zurich)",
    "eu-west-1": "EU (Ireland)",
    "eu-west-2": "EU (London)",
    "eu-west-3": "EU (Paris)",
    "eu-north-1": "EU (Stockholm)",
    "eu-south-1": "Europe (Milan)",
    "eu-south-2": "Europe (Spain)",
    "me-south-1": "Middle East (Bahrain)",
    "me-central-1": "Middle East (UAE)",
    "sa-east-1": "South America (Sao Paulo)",
}


def _clean_text(value, field_name, max_length=100):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required.")
    value = value.strip()
    if len(value) > max_length:
        raise ValueError(f"{field_name} is too long.")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field_name} contains invalid characters.")
    return value


def _normalize_currency(currency):
    normalized = str(currency or "USD").strip().upper()
    if normalized not in SUPPORTED_CURRENCIES:
        raise ValueError(
            "Unsupported currency. Use one of: "
            + ", ".join(sorted(SUPPORTED_CURRENCIES))
        )
    return normalized


def _fetch_json(url, timeout):
    """
    Download and decode a JSON response.

    Some public pricing datasets are delivered with HTTP gzip compression.
    urllib does not consistently decompress these responses automatically,
    so handle both the Content-Encoding header and the gzip magic signature.
    """
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": "CloudMorph-Provider-Pricing/1.3",
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
        content_encoding = str(
            response.headers.get("Content-Encoding", "")
        ).strip().lower()

    # Decompress when the server declares gzip or the payload has gzip magic.
    if content_encoding == "gzip" or payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)

    # utf-8-sig safely handles both regular UTF-8 and UTF-8 BOM responses.
    decoded = payload.decode("utf-8-sig")
    return json.loads(decoded)



def _aws_csv_cache_path(region):
    safe_region = "".join(ch for ch in region.lower() if ch.isalnum() or ch in "-_")
    return os.path.join(AWS_PRICE_CACHE_DIR, f"ec2-{safe_region}-current.csv")


def _download_aws_csv_cached(url, region, timeout):
    """Stream the public regional CSV to disk and reuse it for six hours."""
    os.makedirs(AWS_PRICE_CACHE_DIR, exist_ok=True)
    path = _aws_csv_cache_path(region)
    lock_key = region.lower()

    with _AWS_PRICE_CACHE_LOCK:
        lock = _AWS_PRICE_REGION_LOCKS.setdefault(lock_key, threading.Lock())

    with lock:
        if os.path.isfile(path):
            age = time.time() - os.path.getmtime(path)
            if age < AWS_PRICE_CACHE_TTL_SECONDS and os.path.getsize(path) > 0:
                return path

        temporary_path = path + ".part"
        request = Request(
            url,
            headers={
                "Accept": "text/csv",
                "Accept-Encoding": "identity",
                "User-Agent": "CloudMorph-Provider-Pricing/1.5",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=timeout) as response, open(temporary_path, "wb") as output:
                status = getattr(response, "status", 200)
                if status not in (200,):
                    raise OSError(f"Unexpected AWS CSV HTTP status: {status}")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if os.path.getsize(temporary_path) < 1024:
                raise OSError("AWS regional CSV download was unexpectedly small.")
            os.replace(temporary_path, path)
        except Exception:
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass
            raise
        return path


def _aws_csv_rows(path):
    """Read AWS CSV: first five lines are metadata; sixth line is the header."""
    with open(path, "r", encoding="utf-8-sig", newline="") as source:
        for _ in range(5):
            if not source.readline():
                raise ValueError("AWS CSV metadata/header is incomplete.")
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("AWS CSV header is missing.")
        return reader


def _is_spot_or_low_priority(item):
    searchable = " ".join(
        str(item.get(key) or "")
        for key in ("meterName", "skuName", "productName")
    ).lower()
    return "spot" in searchable or "low priority" in searchable


def _lookup_azure_vm_price_sync(sku, region, currency, os_type, timeout):
    safe_sku = sku.replace("'", "''")
    safe_region = region.replace("'", "''")
    filters = (
        "serviceName eq 'Virtual Machines' "
        f"and armRegionName eq '{safe_region}' "
        f"and armSkuName eq '{safe_sku}' "
        "and priceType eq 'Consumption'"
    )
    query = urlencode({
        "$filter": filters,
        "currencyCode": f"'{currency}'",
    })
    next_url = f"{AZURE_RETAIL_PRICES_ENDPOINT}?{query}"
    items = []
    pages_read = 0

    try:
        while next_url and pages_read < MAX_PAGES:
            data = _fetch_json(next_url, timeout)
            pages_read += 1
            page_items = data.get("Items", [])
            if isinstance(page_items, list):
                items.extend(item for item in page_items if isinstance(item, dict))
            next_url = data.get("NextPageLink")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError,
            json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "provider": "AZURE",
            "reason": "AZURE_PRICING_API_ERROR",
            "message": "Azure retail pricing could not be retrieved.",
            "error_type": type(exc).__name__,
            "pricing_verified": False,
            "source": AZURE_RETAIL_PRICES_SOURCE,
        }

    candidates = []
    for item in items:
        returned_price_type = str(item.get("priceType") or "").strip().lower()
        if returned_price_type and returned_price_type != "consumption":
            continue
        if item.get("armSkuName") != sku:
            continue
        if str(item.get("armRegionName") or "").lower() != region.lower():
            continue
        if _is_spot_or_low_priority(item):
            continue
        unit = str(item.get("unitOfMeasure") or "").strip().lower()
        if unit not in {"1 hour", "1 hours", "hour"}:
            continue
        product_name = str(item.get("productName") or "").lower()
        if os_type == "linux" and "windows" in product_name:
            continue
        if os_type == "windows" and "windows" not in product_name:
            continue
        try:
            hourly_price = float(item.get("retailPrice"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(hourly_price) or hourly_price <= 0:
            continue
        if str(item.get("currencyCode") or "").upper() != currency:
            continue
        candidates.append(item)

    if not candidates:
        return {
            "status": "unavailable",
            "provider": "AZURE",
            "reason": "NO_MATCHING_RETAIL_METER",
            "message": (
                "No matching public Azure Consumption hourly meter was found "
                "for this SKU, region and operating-system selection."
            ),
            "sku": sku,
            "region": region,
            "os_type": os_type,
            "currency": currency,
            "pricing_verified": False,
            "source": AZURE_RETAIL_PRICES_SOURCE,
            "pages_checked": pages_read,
        }

    primary = [
        item for item in candidates
        if item.get("isPrimaryMeterRegion") is True
    ]
    if primary:
        candidates = primary

    distinct = {}
    for item in candidates:
        key = (
            round(float(item["retailPrice"]), 10),
            str(item.get("meterName") or ""),
            str(item.get("productName") or ""),
        )
        distinct[key] = item

    if len(distinct) != 1:
        return {
            "status": "ambiguous",
            "provider": "AZURE",
            "reason": "MULTIPLE_MATCHING_RETAIL_METERS",
            "message": (
                "Azure returned multiple matching retail meters. "
                "CloudMorph will not select one automatically."
            ),
            "sku": sku,
            "region": region,
            "os_type": os_type,
            "currency": currency,
            "matching_meter_count": len(distinct),
            "pricing_verified": False,
            "source": AZURE_RETAIL_PRICES_SOURCE,
            "pages_checked": pages_read,
        }

    selected = next(iter(distinct.values()))
    hourly_price = float(selected["retailPrice"])
    monthly_estimate = round(hourly_price * HOURS_PER_ESTIMATED_MONTH, 2)
    return {
        "status": "success",
        "provider": "AZURE",
        "service": "Virtual Machines",
        "sku": sku,
        "region": region,
        "os_type": os_type,
        "price_type": "Consumption",
        "hourly_price": hourly_price,
        "monthly_compute_estimate": monthly_estimate,
        "annual_compute_estimate": round(monthly_estimate * 12, 2),
        "currency": currency,
        "unit_of_measure": selected.get("unitOfMeasure"),
        "meter_name": selected.get("meterName"),
        "product_name": selected.get("productName"),
        "is_primary_meter_region": selected.get("isPrimaryMeterRegion"),
        "effective_start_date": selected.get("effectiveStartDate"),
        "meter_id": selected.get("meterId"),
        "hours_per_estimated_month": HOURS_PER_ESTIMATED_MONTH,
        "pricing_source": "azure_public_retail_prices_api",
        "pricing_verified": True,
        "verification_scope": (
            "Verified public Azure retail meter for the selected SKU, "
            "region, OS filter and Consumption price type; not verified "
            "against the customer's negotiated contract or invoice."
        ),
        "estimate_scope": "VM compute meter only",
        "excluded_costs": [
            "managed disks and other storage",
            "network bandwidth and data transfer",
            "taxes",
            "support plans",
            "customer-specific discounts or negotiated rates",
            "other Azure services",
        ],
        "source": AZURE_RETAIL_PRICES_SOURCE,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "pages_checked": pages_read,
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
    }


async def get_azure_vm_retail_price(
    sku, region, currency="USD", os_type="linux",
    timeout=DEFAULT_TIMEOUT_SECONDS,
):
    """Retrieve Azure VM Consumption retail pricing for an exact SKU/region."""
    normalized_sku = _clean_text(sku, "sku")
    normalized_region = _clean_text(region, "region").lower()
    normalized_currency = _normalize_currency(currency)
    normalized_os = str(os_type or "linux").strip().lower()
    if normalized_os not in {"linux", "windows"}:
        raise ValueError("os_type must be either 'linux' or 'windows'.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive number.")
    if timeout <= 0 or timeout > 120:
        raise ValueError("timeout must be greater than 0 and at most 120 seconds.")
    return await asyncio.to_thread(
        _lookup_azure_vm_price_sync,
        normalized_sku,
        normalized_region,
        normalized_currency,
        normalized_os,
        float(timeout),
    )


def _extract_aws_hourly_prices(product):
    """Extract positive On-Demand hourly USD prices from one offer product."""
    prices = []
    terms = product.get("terms", {}).get("OnDemand", {})
    for term in terms.values():
        for dimension in term.get("priceDimensions", {}).values():
            unit = str(dimension.get("unit") or "").strip().lower()
            if unit not in {"hrs", "hour", "hours"}:
                continue
            price_map = dimension.get("pricePerUnit", {})
            try:
                value = float(price_map.get("USD"))
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value) and value > 0:
                prices.append(value)
    return prices


def _lookup_aws_ec2_price_sync(instance_type, region, currency, os_type,
                               timeout=DEFAULT_TIMEOUT_SECONDS):
    """Look up AWS EC2 On-Demand price from public CSV without AWS credentials."""
    if currency != "USD":
        return {
            "status": "unavailable", "provider": "AWS",
            "reason": "AWS_PUBLIC_PRICE_LIST_CURRENCY_UNSUPPORTED",
            "message": "The AWS public price-list adapter currently supports USD only. Select USD.",
            "pricing_verified": False, "source": AWS_PRICING_SOURCE,
        }

    location = AWS_REGION_LOCATIONS.get(region.lower())
    if not location:
        return {
            "status": "unavailable", "provider": "AWS",
            "reason": "UNMAPPED_AWS_REGION",
            "message": "This AWS region is not yet supported by the public price-list adapter.",
            "region": region, "pricing_verified": False,
            "source": AWS_PRICING_SOURCE,
        }

    url = f"{AWS_PRICE_LIST_BASE}/{region.lower()}/index.csv"
    try:
        csv_path = _download_aws_csv_cached(url, region, timeout)
        expected_os = "Linux" if os_type == "linux" else "Windows"
        distinct_prices = set()
        publication_date = None
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as source:
            # AWS bulk CSV files document five metadata rows before the header.
            for _ in range(5):
                line = source.readline()
                if not line:
                    raise ValueError("AWS CSV metadata is incomplete.")
                if "publicationDate" in line:
                    publication_date = line.split(",", 1)[-1].strip().strip('"')
            reader = csv.DictReader(source)
            required = {"SKU", "TermType", "Unit", "PricePerUnit", "Location", "Instance Type"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError("AWS CSV does not contain the expected EC2 pricing columns.")

            for row in reader:
                if row.get("TermType") != "OnDemand":
                    continue
                if row.get("Instance Type") != instance_type:
                    continue
                if row.get("Location") != location:
                    continue
                if row.get("Operating System") != expected_os:
                    continue
                if row.get("Tenancy") != "Shared":
                    continue
                if row.get("Pre Installed S/W") not in {"NA", "None", ""}:
                    continue
                if row.get("CapacityStatus") not in {"Used", ""}:
                    continue
                if str(row.get("Unit") or "").strip().lower() not in {"hrs", "hour", "hours"}:
                    continue
                try:
                    price = float(row.get("PricePerUnit"))
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(price) and price > 0:
                    distinct_prices.add(round(price, 10))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, csv.Error) as exc:
        return {
            "status": "unavailable", "provider": "AWS",
            "reason": "AWS_PUBLIC_PRICE_LIST_DOWNLOAD_ERROR",
            "message": "The public AWS EC2 regional CSV price list could not be downloaded or parsed.",
            "error_type": type(exc).__name__, "pricing_verified": False,
            "source": AWS_PRICING_SOURCE, "price_list_url": url,
        }

    if not distinct_prices:
        return {
            "status": "unavailable", "provider": "AWS",
            "reason": "NO_MATCHING_ON_DEMAND_HOURLY_PRICE",
            "message": "No matching AWS EC2 On-Demand hourly USD price was found for this instance type, region and OS.",
            "sku": instance_type, "region": region, "os_type": os_type,
            "currency": currency, "pricing_verified": False,
            "source": AWS_PRICING_SOURCE, "price_list_url": url,
            "price_list_publication_date": publication_date,
        }

    if len(distinct_prices) != 1:
        return {
            "status": "ambiguous", "provider": "AWS",
            "reason": "MULTIPLE_MATCHING_ON_DEMAND_PRICES",
            "message": "AWS public pricing contains multiple matching hourly prices. CloudMorph will not select one automatically.",
            "sku": instance_type, "region": region, "os_type": os_type,
            "currency": currency, "matching_price_count": len(distinct_prices),
            "pricing_verified": False, "source": AWS_PRICING_SOURCE,
            "price_list_url": url,
        }

    hourly_price = next(iter(distinct_prices))
    monthly_estimate = round(hourly_price * HOURS_PER_ESTIMATED_MONTH, 2)
    return {
        "status": "success", "provider": "AWS", "service": "Amazon EC2",
        "sku": instance_type, "region": region, "aws_location": location,
        "os_type": os_type, "price_type": "On-Demand", "tenancy": "Shared",
        "hourly_price": hourly_price,
        "monthly_compute_estimate": monthly_estimate,
        "annual_compute_estimate": round(monthly_estimate * 12, 2),
        "currency": "USD", "unit_of_measure": "Hrs",
        "hours_per_estimated_month": HOURS_PER_ESTIMATED_MONTH,
        "pricing_source": "aws_public_price_list_bulk_csv",
        "pricing_verified": True,
        "verification_scope": (
            "Matched the public AWS EC2 regional CSV offer file for the selected "
            "instance type, region, Linux/Windows OS, shared tenancy and On-Demand term. "
            "This is a public list price, not the customer's actual bill."
        ),
        "estimate_scope": "EC2 instance compute only",
        "excluded_costs": [
            "EBS volumes and snapshots", "data transfer and networking",
            "taxes", "support plans", "customer-specific discounts or Savings Plans",
            "other AWS services",
        ],
        "source": AWS_PRICING_SOURCE, "price_list_url": url,
        "price_list_publication_date": publication_date,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "engine": ENGINE_NAME, "engine_version": ENGINE_VERSION,
    }


async def get_aws_ec2_retail_price(
    instance_type, region, currency="USD", os_type="linux",
    timeout=DEFAULT_TIMEOUT_SECONDS,
):
    """Retrieve public AWS EC2 On-Demand pricing without AWS credentials."""
    normalized_type = _clean_text(instance_type, "instance_type")
    normalized_region = _clean_text(region, "region").lower()
    normalized_currency = _normalize_currency(currency)
    normalized_os = str(os_type or "linux").strip().lower()
    if normalized_os not in {"linux", "windows"}:
        raise ValueError("os_type must be either 'linux' or 'windows'.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive number.")
    if timeout <= 0 or timeout > 120:
        raise ValueError("timeout must be greater than 0 and at most 120 seconds.")
    return await asyncio.to_thread(
        _lookup_aws_ec2_price_sync,
        normalized_type, normalized_region, normalized_currency,
        normalized_os, float(timeout),
    )


def _fetch_gcp_dataset_cached(kind, region, timeout):
    """Fetch public GCP dataset files and cache them for six hours."""
    cache_key = (kind, (region or "").lower())
    now = time.monotonic()
    with _GCP_PUBLIC_DATA_CACHE_LOCK:
        cached = _GCP_PUBLIC_DATA_CACHE.get(cache_key)
        if cached and now - cached["cached_at"] < GCP_PUBLIC_DATA_CACHE_TTL_SECONDS:
            return cached["data"]
    if kind == "specs":
        url = f"{GCP_PUBLIC_DATA_BASE}/specs.json"
    elif kind == "pricing":
        url = f"{GCP_PUBLIC_DATA_BASE}/pricing/{region.lower()}.json"
    else:
        raise ValueError("Unknown GCP dataset kind")
    data = _fetch_json(url, timeout)
    with _GCP_PUBLIC_DATA_CACHE_LOCK:
        _GCP_PUBLIC_DATA_CACHE[cache_key] = {"data": data, "cached_at": time.monotonic()}
    return data


def _gcp_spec_matches(spec, sku):
    if not isinstance(spec, dict):
        return False
    names = (spec.get("type"), spec.get("name"), spec.get("instance_type"), spec.get("instanceType"), spec.get("id"))
    return any(isinstance(value, str) and value.lower() == sku.lower() for value in names)


def _gcp_hourly_price(cell, os_type):
    """Read a public dataset on-demand price across supported schema variants."""
    if not isinstance(cell, dict):
        return None

    def _positive_number(value):
        if isinstance(value, dict):
            # Dataset versions may represent a price as a scalar or an object.
            for key in ("price", "hourly", "value", "amount", "usd"):
                if key in value:
                    result = _positive_number(value[key])
                    if result is not None:
                        return result
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number > 0 else None

    wanted = "linux" if os_type == "linux" else "windows"

    # InstanceFinder documents cells as {instance: {platform: {od, spot, ...}}}.
    # Platform identifiers can include architecture suffixes, so do not rely
    # on a short hard-coded list such as only "linux" or "linux-x86".
    platform_items = []
    for platform_name, platform_cell in cell.items():
        normalized_platform = str(platform_name).strip().lower()
        if wanted in normalized_platform and isinstance(platform_cell, dict):
            platform_items.append((normalized_platform, platform_cell))

    # Prefer an exact platform name, then a platform with an architecture suffix.
    platform_items.sort(key=lambda item: (item[0] != wanted, len(item[0])))
    for _, platform_cell in platform_items:
        if "od" in platform_cell:
            price = _positive_number(platform_cell.get("od"))
            if price is not None:
                return price

    # Backward compatibility with flat cells: {od: ...}.
    if wanted == "linux" and "od" in cell:
        return _positive_number(cell.get("od"))

    return None


def _lookup_gcp_compute_price_sync(machine_type, region, currency, os_type, timeout=DEFAULT_TIMEOUT_SECONDS):
    """Look up GCP machine pricing from a public daily dataset; no GCP credentials."""
    if currency != "USD":
        return {"status": "unavailable", "provider": "GCP", "reason": "GCP_PUBLIC_DATASET_CURRENCY_UNSUPPORTED", "message": "The credential-free GCP public dataset currently provides USD prices only. Select USD.", "pricing_verified": False, "source": GCP_PUBLIC_DATA_SOURCE}
    if os_type != "linux":
        return {"status": "unavailable", "provider": "GCP", "reason": "GCP_OS_NOT_SUPPORTED_BY_DATASET_ADAPTER", "message": "This credential-free GCP adapter currently supports Linux pricing only.", "pricing_verified": False, "source": GCP_PUBLIC_DATA_SOURCE}
    try:
        price_data = _fetch_gcp_dataset_cached("pricing", region, timeout)
        try:
            specs_data = _fetch_gcp_dataset_cached("specs", None, timeout)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            specs_data = []
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "provider": "GCP", "reason": "GCP_PUBLIC_DATASET_DOWNLOAD_ERROR", "message": "The public GCP pricing dataset could not be downloaded. Check backend internet access and try again.", "error_type": type(exc).__name__, "pricing_verified": False, "source": GCP_PUBLIC_DATA_SOURCE}

    specs = specs_data if isinstance(specs_data, list) else specs_data.get("records", specs_data.get("instances", [])) if isinstance(specs_data, dict) else []
    spec = next((item for item in specs if _gcp_spec_matches(item, machine_type)), None)
    if spec is None:
        spec = {}

    # Current InstanceFinder regional files wrap machine prices under
    # the top-level "prices" key and include metadata such as "region"
    # and "generatedAt" beside it. Keep compatibility with older flat
    # and records-wrapped dataset variants as well.
    region_payload = price_data if isinstance(price_data, dict) else {}
    if isinstance(region_payload.get("prices"), dict):
        region_prices = region_payload["prices"]
    elif isinstance(region_payload.get("records"), dict):
        region_prices = region_payload["records"]
    else:
        region_prices = region_payload

    # The dataset identifies its region in the document; do not silently
    # accept a response for a different region.
    returned_region = region_payload.get("region")
    if returned_region and str(returned_region).strip().lower() != region.lower():
        return {"status": "unavailable", "provider": "GCP", "reason": "GCP_DATASET_REGION_MISMATCH", "message": "The public GCP dataset returned a different region than requested.", "requested_region": region, "returned_region": returned_region, "pricing_verified": False, "source": GCP_PUBLIC_DATA_SOURCE}

    price_cell = region_prices.get(machine_type)
    hourly = _gcp_hourly_price(price_cell, os_type)
    if hourly is None:
        return {"status": "unavailable", "provider": "GCP", "reason": "GCP_REGION_OR_PLATFORM_PRICE_NOT_FOUND", "message": "No unambiguous Linux on-demand hourly price was found for this GCP machine type and region in the public dataset.", "sku": machine_type, "region": region, "os_type": os_type, "currency": currency, "pricing_verified": False, "source": GCP_PUBLIC_DATA_SOURCE}

    monthly = round(hourly * HOURS_PER_ESTIMATED_MONTH, 2)
    return {"status": "success", "provider": "GCP", "service": "Compute Engine", "sku": machine_type, "region": region, "os_type": os_type, "price_type": "On-Demand", "hourly_price": hourly, "monthly_compute_estimate": monthly, "annual_compute_estimate": round(monthly * 12, 2), "currency": "USD", "vcpu": spec.get("vcpu", spec.get("vCPU")), "memory_gib": spec.get("memory_gib", spec.get("memory", spec.get("ram_gib"))), "hours_per_estimated_month": HOURS_PER_ESTIMATED_MONTH, "pricing_source": "instancefinder_public_gcp_dataset", "pricing_verified": True, "verification_scope": "Matched the public dataset's GCP machine type and regional Linux on-demand price. Dataset is rebuilt daily from provider public sources; verify against Google Cloud before purchase.", "estimate_scope": "Compute Engine VM compute only", "excluded_costs": ["persistent disks and snapshots", "network egress and other networking", "taxes", "support plans", "committed-use discounts", "customer-specific negotiated rates", "other Google Cloud services"], "source": GCP_PUBLIC_DATA_SOURCE, "dataset_generated_at": (price_data.get("generatedAt") or price_data.get("generated_at")) if isinstance(price_data, dict) else None, "retrieved_at": datetime.now(timezone.utc).isoformat(), "engine": ENGINE_NAME, "engine_version": ENGINE_VERSION}


async def get_gcp_compute_retail_price(machine_type, region, currency="USD", os_type="linux", timeout=DEFAULT_TIMEOUT_SECONDS):
    normalized_type = _clean_text(machine_type, "machine_type")
    normalized_region = _clean_text(region, "region").lower()
    normalized_currency = _normalize_currency(currency)
    normalized_os = str(os_type or "linux").strip().lower()
    if normalized_os not in {"linux", "windows"}:
        raise ValueError("os_type must be either 'linux' or 'windows'.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 120:
        raise ValueError("timeout must be greater than 0 and at most 120 seconds.")
    return await asyncio.to_thread(_lookup_gcp_compute_price_sync, normalized_type, normalized_region, normalized_currency, normalized_os, float(timeout))


async def _fetch_provider_vm_retail_price_uncached(
    provider, sku, region, currency="USD", os_type="linux"
):
    """Dispatch one provider lookup without using the result cache."""
    normalized_provider = str(provider or "").strip().upper()

    if normalized_provider in {"AZURE", "MICROSOFT AZURE"}:
        return await get_azure_vm_retail_price(
            sku=sku, region=region, currency=currency, os_type=os_type
        )

    if normalized_provider in {"AWS", "AMAZON WEB SERVICES"}:
        return await get_aws_ec2_retail_price(
            instance_type=sku,
            region=region,
            currency=currency,
            os_type=os_type,
        )

    if normalized_provider in {"GCP", "GOOGLE CLOUD", "GOOGLE CLOUD PLATFORM"}:
        return await get_gcp_compute_retail_price(
            machine_type=sku,
            region=region,
            currency=currency,
            os_type=os_type,
        )

    return {
        "status": "unavailable",
        "provider": normalized_provider or "UNKNOWN",
        "reason": "UNSUPPORTED_PROVIDER",
        "message": "No pricing adapter is available for this provider.",
        "pricing_verified": False,
    }


async def get_provider_vm_retail_price(
    provider, sku, region, currency="USD", os_type="linux"
):
    """
    Provider-neutral VM pricing entry point with result caching and
    in-flight request de-duplication.

    If several comparison candidates request the same provider/SKU/region
    simultaneously, only one public pricing lookup is performed. The other
    callers await that same result.
    """
    normalized_provider = str(provider or "").strip().upper()
    normalized_sku = _clean_text(sku, "sku")
    normalized_region = _clean_text(region, "region").lower()
    normalized_currency = _normalize_currency(currency)
    normalized_os = str(os_type or "linux").strip().lower()

    if normalized_os not in {"linux", "windows"}:
        raise ValueError("os_type must be either 'linux' or 'windows'.")

    if normalized_provider not in {
        "AZURE", "MICROSOFT AZURE",
        "AWS", "AMAZON WEB SERVICES",
        "GCP", "GOOGLE CLOUD", "GOOGLE CLOUD PLATFORM",
    }:
        return {
            "status": "unavailable",
            "provider": normalized_provider or "UNKNOWN",
            "reason": "UNSUPPORTED_PROVIDER",
            "message": "No pricing adapter is available for this provider.",
            "pricing_verified": False,
        }

    provider_key = (
        "AZURE" if normalized_provider in {"AZURE", "MICROSOFT AZURE"}
        else "AWS" if normalized_provider in {"AWS", "AMAZON WEB SERVICES"}
        else "GCP"
    )
    cache_key = (
        provider_key,
        normalized_sku.lower(),
        normalized_region,
        normalized_currency,
        normalized_os,
    )
    now = time.monotonic()

    async with _VM_PRICE_CACHE_LOCK:
        cached = _VM_PRICE_RESULT_CACHE.get(cache_key)
        if cached and now - cached["cached_at"] < cached["ttl"]:
            return copy.deepcopy(cached["result"])

        task = _VM_PRICE_IN_FLIGHT.get(cache_key)
        if task is None:
            task = asyncio.create_task(
                _fetch_provider_vm_retail_price_uncached(
                    provider=provider_key,
                    sku=normalized_sku,
                    region=normalized_region,
                    currency=normalized_currency,
                    os_type=normalized_os,
                )
            )
            _VM_PRICE_IN_FLIGHT[cache_key] = task
            is_owner = True
        else:
            is_owner = False

    try:
        result = await asyncio.shield(task)
        if is_owner:
            ttl = (
                VM_PRICE_CACHE_TTL_SECONDS
                if isinstance(result, dict) and result.get("status") == "success"
                else VM_PRICE_FAILURE_CACHE_TTL_SECONDS
            )
            async with _VM_PRICE_CACHE_LOCK:
                _VM_PRICE_RESULT_CACHE[cache_key] = {
                    "result": copy.deepcopy(result),
                    "cached_at": time.monotonic(),
                    "ttl": ttl,
                }
        return copy.deepcopy(result)
    finally:
        if is_owner:
            async with _VM_PRICE_CACHE_LOCK:
                if _VM_PRICE_IN_FLIGHT.get(cache_key) is task:
                    _VM_PRICE_IN_FLIGHT.pop(cache_key, None)


def clear_provider_vm_price_cache():
    """Clear in-process exact-price cache; useful for tests and diagnostics."""
    _VM_PRICE_RESULT_CACHE.clear()



def clear_aws_price_list_cache():
    """Delete cached AWS regional CSV offer files; useful for tests and diagnostics."""
    os.makedirs(AWS_PRICE_CACHE_DIR, exist_ok=True)
    with _AWS_PRICE_CACHE_LOCK:
        region_locks = list(_AWS_PRICE_REGION_LOCKS.values())

    # Prevent a cache file from being deleted while its region is downloading.
    for region_lock in region_locks:
        region_lock.acquire()
    try:
        for filename in os.listdir(AWS_PRICE_CACHE_DIR):
            if filename.startswith("ec2-") and (filename.endswith(".csv") or filename.endswith(".part")):
                path = os.path.join(AWS_PRICE_CACHE_DIR, filename)
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
    finally:
        for region_lock in reversed(region_locks):
            region_lock.release()
