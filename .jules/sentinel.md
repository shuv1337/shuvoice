## 2026-02-20 - Insecure IPC Socket Creation
**Vulnerability:** The IPC control socket was created with default permissions (usually 0o755), allowing any local user to control the application.
**Learning:** Unix domain sockets respect the process umask at creation time. `socket.bind()` creates the file.
**Prevention:** Wrap socket binding and directory creation with `os.umask(0o077)` to ensure 0o600/0o700 permissions.
