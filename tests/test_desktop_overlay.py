import unittest

import mimir.desktop as desktop
from mimir.desktop import DesktopWindowController
from mimir.hotkeys import (
    MOD_CONTROL,
    MOD_SHIFT,
    VK_M,
    VK_SPACE,
    audio_hotkey,
    normalize_hotkey_text,
    overlay_hotkey,
    parse_hotkey_text,
)


class FakeWindow:
    def __init__(self) -> None:
        self.visible = False
        self.on_top = True

    def hide(self) -> None:
        self.visible = False

    def show(self) -> None:
        self.visible = True


class DesktopOverlayTests(unittest.TestCase):
    def test_overlay_hotkey_is_show_hide_only(self) -> None:
        spec = overlay_hotkey(lambda: None)

        self.assertEqual(spec.modifiers, MOD_CONTROL)
        self.assertEqual(spec.virtual_key, VK_M)

    def test_audio_hotkey_is_pause_resume_only(self) -> None:
        spec = audio_hotkey(lambda: None)

        self.assertEqual(spec.modifiers, MOD_CONTROL)
        self.assertEqual(spec.virtual_key, VK_SPACE)

    def test_custom_hotkey_text_is_supported(self) -> None:
        modifiers, virtual_key = parse_hotkey_text("Ctrl+Shift+F9")

        self.assertEqual(modifiers, MOD_CONTROL | MOD_SHIFT)
        self.assertEqual(virtual_key, 0x78)

    def test_hotkey_text_is_normalized(self) -> None:
        self.assertEqual(normalize_hotkey_text("control + пробел"), "Ctrl+Space")

    def test_overlay_controller_toggles_window_visibility(self) -> None:
        window = FakeWindow()
        controller = DesktopWindowController(window)

        controller.toggle_overlay()
        self.assertTrue(window.visible)
        self.assertTrue(window.on_top)

        window.on_top = False
        controller.toggle_overlay()
        self.assertFalse(window.visible)

    def test_audio_hotkey_uses_shared_audio_control(self) -> None:
        original_toggle_live_audio = desktop.toggle_live_audio
        calls: list[bool] = []
        desktop.toggle_live_audio = lambda: calls.append(True) or {"running": True}
        try:
            controller = DesktopWindowController(FakeWindow())
            controller.toggle_audio()
        finally:
            desktop.toggle_live_audio = original_toggle_live_audio

        self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()
