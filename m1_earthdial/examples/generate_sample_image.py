"""
Utility script to generate a sample remote sensing image for testing M1 EarthDial.
Simulates a multi-feature satellite image containing agricultural fields, water bodies, and urban structures.
"""

from pathlib import Path
from PIL import Image, ImageDraw


def create_sample_satellite_image(output_path: str = "m1_earthdial/examples/sample_satellite.jpg", size=(512, 512)):
    """Generates a synthetic Earth observation satellite scene for testing."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", size, color=(140, 180, 110))  # Base agricultural green
    draw = ImageDraw.Draw(img)

    # 1. Agricultural parcel grid
    parcels = [
        ((20, 20, 180, 160), (120, 160, 90)),
        ((190, 20, 350, 160), (170, 195, 120)),
        ((360, 20, 490, 160), (160, 140, 90)),
        ((20, 170, 180, 310), (180, 200, 130)),
        ((360, 170, 490, 310), (130, 170, 100)),
    ]
    for box, color in parcels:
        draw.rectangle(box, fill=color, outline=(90, 110, 70), width=2)

    # 2. Meandering river / water body (deep blue)
    river_points = [
        (190, 160), (220, 220), (200, 300), (260, 380), (240, 450), (270, 512),
        (310, 512), (280, 440), (300, 370), (240, 290), (260, 210), (230, 160)
    ]
    draw.polygon(river_points, fill=(35, 85, 145), outline=(25, 65, 110))

    # 3. Urban built-up zone (bottom-left)
    draw.rectangle((20, 340, 210, 490), fill=(160, 160, 165), outline=(100, 100, 105))
    # Buildings within urban area
    buildings = [
        ((35, 355, 75, 395), (210, 80, 60)),    # Red rooftop
        ((90, 355, 140, 395), (220, 220, 225)), # White warehouse
        ((150, 355, 195, 395), (70, 90, 130)),  # Commercial blue roof
        ((35, 415, 85, 470), (215, 140, 70)),   # Industrial roof
        ((100, 415, 145, 470), (200, 75, 55)),  # Residential block
        ((155, 415, 195, 470), (225, 225, 230)),# Concrete structure
    ]
    for b_box, b_color in buildings:
        draw.rectangle(b_box, fill=b_color, outline=(50, 50, 50), width=1)

    # 4. Transportation / Road networks (asphalt gray)
    draw.line([(0, 330), (512, 330)], fill=(80, 80, 85), width=6)  # East-West highway
    draw.line([(185, 0), (185, 512)], fill=(80, 80, 85), width=4)  # North-South arterial road

    img.save(path, quality=95)
    print(f"[+] Successfully generated sample satellite image: {path.resolve()}")
    return path


if __name__ == "__main__":
    create_sample_satellite_image()

