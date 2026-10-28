"""Show the bot's status on a small OLED screen (128 x 64 pixels, I2C; SSD1306 or SH1106 chip).

    python oled.py            draw the current status on the screen once
    python oled.py preview    no screen needed: save what it would show to state/oled_preview.png
run.py calls show() after every cycle, and ignores it quietly if no screen is attached.
On the Pi:  pip install luma.oled    and turn I2C on with  sudo raspi-config  (Interface Options -> I2C)
"""
import os, sys

from PIL import Image, ImageDraw, ImageFont

import config as C

W, H = 128, 64
DRIVER = os.environ.get("OLED_DRIVER", "ssd1306")              # or sh1106 (most 1.3 inch screens)
ADDRESS = int(os.environ.get("OLED_ADDRESS", "0x3C"), 16)


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def frame(d):
    """Draw the status dictionary (from status.summary) as a 128 x 64 black-and-white image."""
    img = Image.new("1", (W, H), 0); g = ImageDraw.Draw(img)
    big, small = _font(16), _font(10)
    ok = d["status"] == "OK"
    if ok:
        g.text((0, 0), d["position"], font=big, fill=1)
    else:                                                       # inverted bar: something needs attention
        g.rectangle((0, 0, W - 1, 17), fill=1); g.text((2, 0), d["status"], font=big, fill=0)
    if d.get("price"):
        p = f"{d['price']:,.0f}"
        g.text((W - g.textlength(p, font=big), 0), p, font=big, fill=0 if not ok else 1)
    g.line((0, 20, W - 1, 20), fill=1)
    right = lambda y, t: g.text((W - g.textlength(t, font=small), y), t, font=small, fill=1)
    if "live" in d:
        g.text((0, 22), f"match {d['matched']}/{d['live']}", font=small, fill=1); right(22, f"miss {d['missed']}")
        g.text((0, 34), f"paper {d['eq_live'] * 100:+.2f}%", font=small, fill=1); right(34, f"halt {d['halted']}")
    note = (d.get("note") or "").split(":")[0]
    g.text((0, 50), note or "paper only", font=small, fill=1); right(50, d["last_run"][-5:] + "Z")
    return img


def show(d):
    """Send the frame to the screen. Returns False (and does nothing) if no screen or library is present."""
    try:
        from luma.core.interface.serial import i2c
        from luma.oled import device as dev
        getattr(dev, DRIVER)(i2c(port=1, address=ADDRESS)).display(frame(d))
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import status
    from store import Store
    d = status.summary(Store(C.DB_PATH))
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        out = os.path.join(C.HERE, "state", "oled_preview.png")
        frame(d).resize((W * 4, H * 4)).save(out); print("saved", out)
    else:
        print("drawn on the screen" if show(d) else "no screen found (is I2C on, and luma.oled installed?)")
