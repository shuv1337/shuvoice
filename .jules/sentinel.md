## 2026-02-20 - Insecure IPC Socket Creation
**Vulnerability:** The IPC control socket was created with default permissions (usually 0o755), allowing any local user to control the application.
**Learning:** Unix domain sockets respect the process umask at creation time. `socket.bind()` creates the file.
**Prevention:** Wrap socket binding and directory creation with `os.umask(0o077)` to ensure 0o600/0o700 permissions.

## 2026-02-20 - Pre-created Directory Hijack Risk
**Vulnerability:** Control socket creation in shared locations (/tmp) did not verify directory ownership, allowing attackers to pre-create the directory.
**Learning:** Even with `umask`, `mkdir` fails silently if the directory exists. Directory ownership must be verified explicitly.
**Prevention:** Check `path.stat().st_uid == os.getuid()` before using a directory in shared paths.
