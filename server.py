"""
Google Vision OCR MCP Server
Wraps Google Cloud Vision API for text extraction from images.
Supports both base64-encoded images and public image URLs.
"""

import base64
import json
import os
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Constants ────────────────────────────────────────────────────────────────

GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")

# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP("google_vision_ocr_mcp")

# ── Pydantic models ───────────────────────────────────────────────────────────

class OCRFromBase64Input(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_base64: str = Field(
        ...,
        description=(
            "Base64-encoded image content (JPEG, PNG, GIF, BMP, WEBP, RAW, ICO, PDF, TIFF). "
            "Do NOT include the data-URI prefix (e.g. 'data:image/png;base64,') — "
            "pass only the raw base64 string."
        ),
        min_length=10,
    )
    language_hints: Optional[list[str]] = Field(
        default=None,
        description=(
            "BCP-47 language hints to improve recognition accuracy. "
            "Examples: ['ru'], ['kk'], ['en'], ['ru', 'kk']. "
            "Leave empty for auto-detection."
        ),
        max_length=5,
    )

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        # Strip data-URI prefix if user accidentally included it
        if "," in v and v.startswith("data:"):
            v = v.split(",", 1)[1]
        # Validate it is valid base64
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError(
                "image_base64 is not valid base64. "
                "Encode the raw image bytes with base64.b64encode() and pass the result."
            )
        return v


class OCRFromURLInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    image_url: str = Field(
        ...,
        description=(
            "Publicly accessible URL of the image. "
            "Must start with https:// or http://. "
            "The URL must be reachable by Google's servers."
        ),
        min_length=10,
    )
    language_hints: Optional[list[str]] = Field(
        default=None,
        description=(
            "BCP-47 language hints. Examples: ['ru'], ['kk', 'ru'], ['en']. "
            "Leave empty for auto-detection."
        ),
        max_length=5,
    )

    @field_validator("image_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("image_url must start with http:// or https://")
        return v


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_request_body(image_source: dict, language_hints: Optional[list[str]]) -> dict:
    """Construct the Vision API request payload."""
    context = {}
    if language_hints:
        context["languageHints"] = language_hints

    feature = {"type": "TEXT_DETECTION", "maxResults": 1}

    return {
        "requests": [
            {
                "image": image_source,
                "features": [feature],
                **({"imageContext": context} if context else {}),
            }
        ]
    }


def _handle_vision_response(response_data: dict) -> str:
    """Parse Vision API response and return extracted text or error."""
    responses = response_data.get("responses", [])
    if not responses:
        return json.dumps({"error": "Vision API returned an empty response."})

    resp = responses[0]

    # API-level error
    if "error" in resp:
        code = resp["error"].get("code", "unknown")
        message = resp["error"].get("message", "Unknown error")
        return json.dumps({
            "error": f"Vision API error {code}: {message}",
            "suggestion": (
                "Check that your GOOGLE_VISION_API_KEY is valid and that "
                "Cloud Vision API is enabled in your GCP project."
            ),
        })

    # No text found
    full_annotation = resp.get("fullTextAnnotation")
    if not full_annotation:
        return json.dumps({
            "text": "",
            "message": "No text was detected in the image.",
        })

    extracted_text = full_annotation.get("text", "")

    # Collect per-page confidence if available
    pages_info = []
    for page in full_annotation.get("pages", []):
        confidence = page.get("confidence")
        if confidence is not None:
            pages_info.append(round(confidence, 4))

    result = {"text": extracted_text}
    if pages_info:
        result["confidence_per_page"] = pages_info

    return json.dumps(result, ensure_ascii=False, indent=2)


def _handle_http_error(e: Exception) -> str:
    """Consistent HTTP error formatting."""
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 400:
            return json.dumps({"error": "Bad request — check image format or API key."})
        if status == 403:
            return json.dumps({
                "error": "Permission denied (403).",
                "suggestion": "Verify GOOGLE_VISION_API_KEY and that Cloud Vision API is enabled.",
            })
        if status == 429:
            return json.dumps({
                "error": "Rate limit exceeded (429). Wait before retrying.",
            })
        return json.dumps({"error": f"HTTP {status}: {e.response.text[:300]}"})
    if isinstance(e, httpx.TimeoutException):
        return json.dumps({"error": "Request timed out. The image may be too large or the network slow."})
    return json.dumps({"error": f"Unexpected error: {type(e).__name__}: {e}"})


async def _call_vision_api(body: dict) -> str:
    """Make the actual async HTTP call to Google Vision API."""
    if not API_KEY:
        return json.dumps({
            "error": "GOOGLE_VISION_API_KEY environment variable is not set.",
            "suggestion": "Set it in your Railway environment variables.",
        })

    url = f"{GOOGLE_VISION_URL}?key={API_KEY}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=body)
        response.raise_for_status()
        return _handle_vision_response(response.json())


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="ocr_from_base64",
    annotations={
        "title": "OCR from Base64 Image",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ocr_from_base64(params: OCRFromBase64Input) -> str:
    """Extract all text from a base64-encoded image using Google Cloud Vision API.

    Supports JPEG, PNG, GIF, BMP, WEBP, PDF, TIFF.
    Use this when you have the raw image bytes (e.g. uploaded file).

    Args:
        params (OCRFromBase64Input):
            - image_base64 (str): Raw base64-encoded image, no data-URI prefix.
            - language_hints (list[str] | None): BCP-47 codes e.g. ['ru', 'kk'].

    Returns:
        str: JSON with fields:
            - text (str): Full extracted text.
            - confidence_per_page (list[float]): Optional per-page confidence scores.
            - error (str): Present only on failure.
    """
    try:
        image_source = {"content": params.image_base64}
        body = _build_request_body(image_source, params.language_hints)
        return await _call_vision_api(body)
    except httpx.HTTPStatusError as e:
        return _handle_http_error(e)
    except Exception as e:
        return _handle_http_error(e)


@mcp.tool(
    name="ocr_from_url",
    annotations={
        "title": "OCR from Image URL",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def ocr_from_url(params: OCRFromURLInput) -> str:
    """Extract all text from a publicly accessible image URL using Google Cloud Vision API.

    The URL must be reachable by Google's servers (public, no auth required).

    Args:
        params (OCRFromURLInput):
            - image_url (str): Public https:// or http:// URL of the image.
            - language_hints (list[str] | None): BCP-47 codes e.g. ['ru', 'en'].

    Returns:
        str: JSON with fields:
            - text (str): Full extracted text.
            - confidence_per_page (list[float]): Optional per-page confidence scores.
            - error (str): Present only on failure.
    """
    try:
        image_source = {"source": {"imageUri": params.image_url}}
        body = _build_request_body(image_source, params.language_hints)
        return await _call_vision_api(body)
    except httpx.HTTPStatusError as e:
        return _handle_http_error(e)
    except Exception as e:
        return _handle_http_error(e)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="streamable_http", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
