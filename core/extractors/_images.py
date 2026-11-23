"""Read bounded screenshot and QR images into memory."""

from io import BytesIO

from PIL import Image, ImageOps

from core.validators import MAX_IMAGE_BYTES


MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_SIDE = 8192
_FORMATS = ("PNG", "JPEG", "WEBP", "BMP", "GIF", "TIFF")


class ImageInputError(ValueError):
    """The input is not a supported, bounded still image."""


def load_image(data: bytes) -> Image.Image:
    """Return an oriented RGB image with transparency composited onto white."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(f"Provide a nonempty image of at most {MAX_IMAGE_BYTES:,} bytes.")

    try:
        with Image.open(BytesIO(data), formats=_FORMATS) as original:
            width, height = original.size
            if max(width, height) > MAX_IMAGE_SIDE or width * height > MAX_IMAGE_PIXELS:
                raise ImageInputError("The image dimensions exceed the processing limit.")
            if getattr(original, "is_animated", False):
                raise ImageInputError("Provide a single-frame image.")

            image = ImageOps.exif_transpose(original)
            if "A" in image.getbands() or "transparency" in image.info:
                rgba = image.convert("RGBA")
                white = Image.new("RGB", image.size, "white")
                white.paste(rgba, mask=rgba.getchannel("A"))
                return white
            return image.convert("RGB")
    except ImageInputError:
        raise
    except Image.DecompressionBombError:
        raise ImageInputError("The image dimensions exceed the processing limit.") from None
    except (OSError, SyntaxError, ValueError):
        raise ImageInputError("The image could not be read as a supported format.") from None
