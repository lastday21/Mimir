import unittest

from mimir.desktop import DesktopWindowController
from mimir.hotkeys import MOD_CONTROL, VK_M, VK_SPACE, audio_hotkey, overlay_hotkey


class FakeWindow:
    def __init__(self) -> None:
        self.visible = True
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

    def test_overlay_controller_toggles_window_visibility(self) -> None:
        window = FakeWindow()
        controller = DesktopWindowController(window)

        controller.toggle_overlay()
        self.assertFalse(window.visible)

        window.on_top = False
        controller.toggle_overlay()
        self.assertTrue(window.visible)
        self.assertTrue(window.on_top)


if __name__ == "__main__":
    unittest.main()
