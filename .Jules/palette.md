## 2026-02-28 - Focus States for Custom GTK Buttons
**Learning:** When adding custom CSS classes to interactive GTK4 widgets like Gtk.Button (e.g., .wizard-btn), they can lose their default visual focus ring during keyboard navigation (tabbing). This makes the interface inaccessible for keyboard users, as they cannot see which element is currently active.
**Action:** Always ensure custom button classes define a explicit `:focus-visible` pseudo-class style (e.g., using `outline` and `outline-offset`) to maintain keyboard accessibility.
